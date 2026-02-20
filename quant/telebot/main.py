
import logging
import os
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from quant.telebot.models import Base, User, UserContext
from quant.telebot.auth import CryptoManager
from quant.telebot.manager import BotManager

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

FOOTER = "\n\nℹ️ Run /help to see command list"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_model(root: Path = Path("models/production")) -> Path | None:
    if not root.exists():
        return None
    subdirs = sorted([x for x in root.iterdir() if x.is_dir() and "model_" in x.name], key=lambda x: x.name)
    if subdirs:
        return subdirs[-1]
    return None

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
# find model
MODEL_DIR = find_latest_model()
if not MODEL_DIR:
    logger.warning("No production model found in models/production! Bot will fail to start trading.")
else:
    logger.info(f"Using latest model: {MODEL_DIR}")

# Db - StaticPool + check_same_thread=False for SQLite in async context
DB_PATH = os.path.abspath("quant_bot.db")
ENGINE = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=ENGINE)
Base.metadata.create_all(ENGINE)

# Enable WAL mode for better concurrent read/write
with ENGINE.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.commit()

# Crypto
# Ensure key exists or fail fast
if not os.getenv("BOT_MASTER_KEY"):
    # Generate one for dev convenience if missing, but warn
    k = CryptoManager.generate_key()
    logger.warning(f"BOT_MASTER_KEY not set! Using temporary key: {k}")
    # In prod, this would be an error. For now, we set it so auth works for this session.
    os.environ["BOT_MASTER_KEY"] = k

CRYPTO = CryptoManager()

# Manager
MANAGER = BotManager(MODEL_DIR) if MODEL_DIR else None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    session = SessionLocal()
    try:
        db_user = session.query(User).filter_by(telegram_id=user.id).first()
        
        if not db_user:
            # Register new user
            new_user = User(
                telegram_id=user.id,
                username=user.username,
                role='user',
                status='pending'
            )
            context_rec = UserContext(telegram_id=user.id)
            new_user.context = context_rec
            session.add(new_user)
            session.commit()
            
            await update.message.reply_text(
                f"👋 Welcome {user.first_name}!\n\n"
                "⏳ **Account Pending**\n"
                "Your request has been sent to the administrator.\n\n"
                "👉 **Next Step:** Wait for approval notification.\n"
                "_(You will be notified here automatically)_" + FOOTER
            )
            
            # Notify Admin
            admin_id = os.getenv("ADMIN_ID")
            if admin_id:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🔔 **New User:** {user.first_name} (@{user.username}) [ID: {user.id}]\n"
                             f"👉 Run `/approve {user.id}` to grant access."
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin: {e}")
        elif db_user.status == 'pending':
            await update.message.reply_text("⏳ Request still pending. Please wait for admin approval." + FOOTER)
        elif db_user.status == 'banned':
            await update.message.reply_text("🚫 Access denied. Contact admin." + FOOTER)
        else:
            await update.message.reply_text(
                f"✅ **Welcome back, {user.first_name}!**\n\n"
                "System is ready.\n"
                "👉 **Next Step:** Check status or start trading.\n"
                "Run: `/status`, `/start_demo`, or `/start_live`" + FOOTER
            )
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("⚠️ System error. Try again later.")
    finally:
        session.close()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"HELP COMMAND CALLED by {update.effective_user.id}")
    try:
        msg = (
            "COMMAND LIST\n\n"
            "Basics\n"
            "/start - Check account status\n"
            "/help - Show this menu\n\n"
            "Setup\n"
            "/setup email key pass - Connect Capital.com\n"
            "  (no brackets - just paste values separated by spaces)\n\n"
            "Trading\n"
            "/start_demo - Start PAPER trading\n"
            "/start_live - Start REAL trading\n"
            "/stop - Stop execution\n"
            "/status - Check if running\n"
            "/stats - View live performance\n\n"
            "NOTE: Capital.com demo and live use separate API keys.\n"
            "Create your API key inside the demo or live platform\n"
            "matching the mode you want to trade in."
        )
        
        user_id = update.effective_user.id
        admin_id_str = os.getenv("ADMIN_ID", "")
        if str(user_id) == admin_id_str:
            msg += "\n\nAdmin\n/approve <id> - Approve user\n/revoke <id> - Freeze user"
        
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"CRITICAL HELP ERROR: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error displaying help menu.")

async def _start_engine(update: Update, context: ContextTypes.DEFAULT_TYPE, live: bool):
    if not MANAGER: return
    user_id = update.effective_user.id
    
    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()
    
    if not user or user.status != 'active':
        await update.message.reply_text("⛔ Account not approved." + FOOTER)
        session.close()
        return
        
    ctx = user.context
    if not (ctx.capital_email and ctx.capital_api_key and ctx.capital_password):
        await update.message.reply_text("❌ Credentials missing. Run /setup first." + FOOTER)
        session.close()
        return
        
    # Decrypt
    try:
        creds = {
            'email': ctx.capital_email,
            'api_key': CRYPTO.decrypt(ctx.capital_api_key),
            'password': CRYPTO.decrypt(ctx.capital_password),
            'live': live
        }
    except Exception as e:
        logger.error(f"Decryption failed for user {user_id}: {e}")
        await update.message.reply_text("❌ Decryption failed. Re-run /setup." + FOOTER)
        session.close()
        return
        
    # Update preference
    if ctx.live_mode != live:
        ctx.live_mode = live
        session.commit()
        
    session.close()

    mode_str = "LIVE 🔴" if live else "DEMO 🟢"
    
    async def notify_signal(result):
        try:
            signal_type = result.get('signal', 'HOLD')
            if signal_type == 'HOLD':
                return  # Don't spam user with HOLD signals
            emoji = {"BUY": "🟢", "SELL": "🔴"}.get(signal_type, "❓")
            msg = (
                f"🔔 **Trading Signal**\n\n"
                f"{emoji} Action: **{signal_type}**\n"
                f"Price: `{result.get('close_price', 0.0):.5f}`\n"
                f"Regime: {result.get('regime', '?')} | P(up): {result.get('probability', 0.0):.3f}\n"
                f"Reason: {result.get('reason', 'N/A')}"
            )
            if result.get('position'):
                pos = result['position']
                msg += f"\nSize: {pos.get('lot_size', 0):.2f} lots | Risk: {pos.get('risk_fraction', 0)*100:.1f}%"
            await context.bot.send_message(chat_id=user_id, text=msg)
        except Exception as e:
            logger.error(f"Failed to send signal notification to user {user_id}: {e}")

    try:
        started = await MANAGER.start_session(user_id, creds, on_signal=notify_signal)
        if started:
            await update.message.reply_text(
                f"🚀 **{mode_str} Trading STARTED**\n\n"
                "✅ Analysis running...\n"
                "👉 **Next Step:** Monitor performance.\n"
                "Run: `/stats`" + FOOTER
            )
        else:
            await update.message.reply_text("⚠️ Engine already running." + FOOTER)
    except Exception as e:
        logger.error(f"Failed to start engine for user {user_id}: {e}")
        await update.message.reply_text(f"❌ Failed to start engine: {e}" + FOOTER)

async def start_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=False)

async def start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=True)

async def stop_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER: return
    user_id = update.effective_user.id
    
    if await MANAGER.stop_session(user_id):
        await update.message.reply_text(
            "Bzzt. **Engine STOPPED** 🛑\n\n"
            "To resume:\n"
            "`/start_demo` or `/start_live`" + FOOTER
        )
    else:
        await update.message.reply_text("⚠️ Engine not running." + FOOTER)


async def revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    admin_id = os.getenv("ADMIN_ID")
    
    if str(user_id) != str(admin_id):
        return

    try:
        target_id = int(context.args[0])
        session = SessionLocal()
        user = session.query(User).filter_by(telegram_id=target_id).first()
        if user:
            user.status = 'banned'
            session.commit()
            
            # Stop engine if running
            if MANAGER and MANAGER.is_running(target_id):
                await MANAGER.stop_session(target_id)
                await update.message.reply_text(f"🛑 active session for {target_id} stopped.")
                
            await update.message.reply_text(f"🚫 User {target_id} has been **FROZEN**." + FOOTER)

            try:
                await context.bot.send_message(target_id, "⛔ Your access has been revoked by the administrator.")
            except:
                pass 
        else:
            await update.message.reply_text("❌ User not found." + FOOTER)
        session.close()
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /revoke <user_id>" + FOOTER)

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    admin_id = os.getenv("ADMIN_ID")
    
    if str(user_id) != str(admin_id):
        return

    try:
        target_id = int(context.args[0])
        session = SessionLocal()
        user = session.query(User).filter_by(telegram_id=target_id).first()
        if user:
            user.status = 'active'
            session.commit()
            await update.message.reply_text(f"✅ User {target_id} has been **APPROVED**." + FOOTER)
            try:
                await context.bot.send_message(target_id, "🎉 **Account Approved!**\n\nYou can now set up your trading credentials.\nRun: `/setup your@email.com YOUR_API_KEY YOUR_PASSWORD`\n(no brackets - just paste values directly)")
            except:
                pass
        else:
            await update.message.reply_text("❌ User not found." + FOOTER)
        session.close()
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /approve <user_id>" + FOOTER)

async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) < 3:
        await update.message.reply_text(
            "⚠️ Usage:\n"
            "`/setup your@email.com YOUR_API_KEY YOUR_PASSWORD`\n\n"
            "⚠️ Do NOT include < > brackets — just paste the values directly.\n\n"
            "**Security Note:** Your keys are encrypted." + FOOTER
        )
        return

    email = args[0]
    api_key = args[1]
    password = args[2] # In reality, user might send multiple words, but Capital passwords usually don't have spaces or we assume last arg
    
    # Encrypt
    enc_key = CRYPTO.encrypt(api_key)
    enc_pass = CRYPTO.encrypt(password)
    
    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()
    if user:
        if not user.context:
            user.context = UserContext(telegram_id=user_id)
        
        user.context.capital_email = email
        user.context.capital_api_key = enc_key
        user.context.capital_password = enc_pass
        session.commit()
        await update.message.reply_text("✅ **Credentials Saved!**\n\nYou can now start trading.\nRun: `/start_demo` or `/start_live`" + FOOTER)
    else:
        await update.message.reply_text("⛔ User not found." + FOOTER)
    session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER: return
    user_id = update.effective_user.id
    
    if not MANAGER.is_running(user_id):
        await update.message.reply_text("⚠️ Engine not running." + FOOTER)
        return
        
    engine = MANAGER.sessions[user_id]
    gen = engine.gen

    # Determine mode
    base_url = gen.client._cfg.base_url.lower()
    mode_label = "LIVE 🔴" if "api-capital" in base_url and "demo" not in base_url else "DEMO 🟢"

    # Win rate stats
    wr = gen.get_win_rate_stats()

    # Recent signals (always available even if API is down)
    recent = gen.signal_log[-5:] if gen.signal_log else []
    if recent:
        trade_lines = []
        for s in recent:
            outcome = s.get('outcome', '')
            tag = ""
            if outcome == "win":
                tag = " W"
            elif outcome == "loss":
                tag = " L"
            trade_lines.append(
                f"- {s['signal']} @ {s['close_price']:.5f} (P={s['probability']:.3f}){tag}"
            )
        trade_str = "\n".join(trade_lines)
    else:
        trade_str = "(No signals yet)"

    # Try to fetch live account data
    balance_str = "N/A"
    pnl_str = "N/A"
    pos_count = "N/A"
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gen._ensure_authenticated)
        acct = await loop.run_in_executor(None, gen.client.get_accounts)
        positions = await loop.run_in_executor(None, gen.client.get_positions)

        balance = acct.get('balance', {}).get('balance', 0) if isinstance(acct.get('balance'), dict) else acct.get('balance', 0)
        total_pnl = sum([p.get('position', {}).get('profit', 0) for p in positions])
        balance_str = f"${balance:,.2f}"
        pnl_str = f"${total_pnl:,.2f}"
        pos_count = str(len(positions))
    except Exception as e:
        logger.warning(f"Could not fetch account data: {e}")

    # Build win rate section
    if wr["evaluated"] > 0:
        wr_section = (
            f"\n**Win Rate**\n"
            f"Win Rate: {wr['win_rate']}% ({wr['wins']}W / {wr['losses']}L)\n"
            f"Total Pips: {wr['total_pips']:+.1f}\n"
            f"Evaluated: {wr['evaluated']} | Pending: {wr['pending']}\n"
        )
    else:
        pending = wr['pending']
        if pending > 0:
            wr_section = f"\n**Win Rate**\n(Evaluating {pending} signal(s)...)\n"
        else:
            wr_section = ""

    msg = (
        f"📊 **{mode_label} Statistics**\n\n"
        f"💰 **Balance:** {balance_str}\n"
        f"📈 **PnL (Open):** {pnl_str}\n"
        f"🟢 **Positions:** {pos_count}\n"
        f"{wr_section}\n"
        f"**Signals:** {wr['total_signals']} total "
        f"({wr['buys']} BUY / {wr['sells']} SELL / {wr['holds']} HOLD)\n\n"
        f"**Recent:**\n{trade_str}" +
        FOOTER
    )
    await update.message.reply_text(msg)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER: 
        await update.message.reply_text("❌ Bot Manager not initialized." + FOOTER)
        return
        
    user_id = update.effective_user.id
    if MANAGER.is_running(user_id):
        await update.message.reply_text("✅ **Engine Running** 🏃\n\nMonitoring market..." + FOOTER)
    else:
        await update.message.reply_text("🛑 **Engine Stopped**" + FOOTER)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("Error: TELEGRAM_TOKEN environment variable is missing.")
        return
        
    application = ApplicationBuilder().token(token).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('menu', help_command))
    application.add_handler(CommandHandler('commands', help_command))
    
    application.add_handler(CommandHandler('approve', approve))
    application.add_handler(CommandHandler('revoke', revoke))
    application.add_handler(CommandHandler('setup', setup))
    application.add_handler(CommandHandler('start_demo', start_demo))
    application.add_handler(CommandHandler('star_demo', start_demo))
    application.add_handler(CommandHandler('start_live', start_live))
    application.add_handler(CommandHandler('star_live', start_live))
    application.add_handler(CommandHandler('stop', stop_trading))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('stats', stats))
    
    # Debug: Log all updates
    async def debug_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"DEBUG: Received update: {update}")
    
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.ALL, debug_log), group=1)
    
    application.add_error_handler(error_handler)
    
    print("Bot is polling...")
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
