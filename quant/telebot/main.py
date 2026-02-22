
import logging
import os
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


def _persist_user_session_flags(
    user_id: int,
    *,
    is_active: bool | None = None,
    live_mode: bool | None = None,
) -> None:
    """Persist session continuity flags for a user context."""
    session = SessionLocal()
    try:
        db_user = session.query(User).filter_by(telegram_id=user_id).first()
        if not db_user:
            return

        db_ctx = db_user.context
        changed = False
        if not db_ctx:
            db_ctx = UserContext(telegram_id=user_id)
            db_user.context = db_ctx
            changed = True

        if live_mode is not None and db_ctx.live_mode != live_mode:
            db_ctx.live_mode = live_mode
            changed = True
        if is_active is not None and db_ctx.is_active != is_active:
            db_ctx.is_active = is_active
            changed = True

        if changed:
            session.commit()
    except Exception as e:
        session.rollback()
        logger.warning(f"Failed to persist session flags for user {user_id}: {e}")
    finally:
        session.close()


def _build_creds_from_context(ctx: UserContext | None, *, live: bool) -> dict:
    """Build credential payload expected by BotManager.start_session()."""
    binance_key = CRYPTO.decrypt(ctx.binance_api_key) if ctx and ctx.binance_api_key else ""
    binance_secret = CRYPTO.decrypt(ctx.binance_api_secret) if ctx and ctx.binance_api_secret else ""

    if live and (not binance_key or not binance_secret):
        raise RuntimeError("Binance API credentials required for live trading")

    return {
        "live": live,
        "binance_api_key": binance_key,
        "binance_api_secret": binance_secret,
    }


def _build_signal_notifier(bot, user_id: int):
    """Create the async callback used by the engine to notify signals."""

    async def notify_signal(result):
        try:
            signal_type = result.get("signal", "HOLD")

            # Handle engine crash notification
            if signal_type == "ENGINE_CRASH":
                reason = result.get("reason", "Unknown error")
                msg = (
                    f"⚠️ **ENGINE CRASHED**\n\n"
                    f"Reason: {reason}\n\n"
                    f"The engine has stopped. Use /start_demo or /start_live to restart."
                )
                await bot.send_message(chat_id=user_id, text=msg)
                _persist_user_session_flags(user_id, is_active=False)

                # Clean up dead session
                if MANAGER and user_id in MANAGER.sessions:
                    del MANAGER.sessions[user_id]
                return

            if signal_type == "HOLD":
                return  # Don't spam user with HOLD signals
            emoji = {"BUY": "🟢", "SELL": "🔴"}.get(signal_type, "❓")
            msg = (
                f"🔔 **Trading Signal**\n\n"
                f"{emoji} Action: **{signal_type}**\n"
                f"Price: `{result.get('close_price', 0.0):.5f}`\n"
                f"Regime: {result.get('regime', '?')} | P(up): {result.get('probability', 0.0):.3f}\n"
                f"Reason: {result.get('reason', 'N/A')}"
            )
            if result.get("position"):
                pos = result["position"]
                msg += f"\nSize: {pos.get('lot_size', 0):.2f} lots | Risk: {pos.get('risk_fraction', 0)*100:.1f}%"
            await bot.send_message(chat_id=user_id, text=msg)
        except Exception as e:
            logger.error(f"Failed to send signal notification to user {user_id}: {e}")

    return notify_signal


async def _restore_active_sessions(application):
    """Restore active sessions from DB on bot startup."""
    if not MANAGER:
        return

    session = SessionLocal()
    restore_targets: list[tuple[int, bool, dict]] = []
    try:
        active_users = (
            session.query(User)
            .join(UserContext, UserContext.telegram_id == User.telegram_id)
            .filter(User.status.in_(("active", "approved")), UserContext.is_active.is_(True))
            .all()
        )

        for db_user in active_users:
            ctx = db_user.context
            if not ctx:
                continue

            live = bool(ctx.live_mode)
            try:
                creds = _build_creds_from_context(ctx, live=live)
            except Exception as cred_err:
                logger.warning(f"Skipping auto-restore for user {db_user.telegram_id}: {cred_err}")
                ctx.is_active = False
                continue

            restore_targets.append((db_user.telegram_id, live, creds))

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed loading sessions for auto-restore: {e}", exc_info=True)
        return
    finally:
        session.close()

    if not restore_targets:
        logger.info("No persisted active sessions to restore.")
        return

    logger.info("Restoring %d persisted session(s)...", len(restore_targets))
    for user_id, live, creds in restore_targets:
        try:
            started = await MANAGER.start_session(
                user_id,
                creds,
                on_signal=_build_signal_notifier(application.bot, user_id),
            )
            if started:
                _persist_user_session_flags(user_id, is_active=True, live_mode=live)
                logger.info("Restored session for user %s in %s mode.", user_id, "LIVE" if live else "DEMO")
                try:
                    await application.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "♻️ **Session Restored**\n\n"
                            "System restarted, and your trading session resumed automatically."
                            + FOOTER
                        ),
                    )
                except Exception as notify_err:
                    logger.warning(f"Failed to notify restored session for user {user_id}: {notify_err}")
        except Exception as restore_err:
            logger.error(f"Failed to restore session for user {user_id}: {restore_err}")
            _persist_user_session_flags(user_id, is_active=False)
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text=(
                        "⚠️ Could not auto-resume your trading session.\n"
                        "Please run `/start_demo` or `/start_live`."
                        + FOOTER
                    ),
                )
            except Exception:
                pass


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
                "👉 `/start_demo` — paper trading (no API keys needed)\n"
                "👉 `/start_live` — real trading (requires `/setup` first)\n"
                "👉 `/status` — check engine state" + FOOTER
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
            "/setup key secret - Connect Binance (optional for paper)\n"
            "  Paper trading works without credentials!\n\n"
            "Trading\n"
            "/start_demo - Start PAPER trading (BTCUSDT 1H)\n"
            "/start_live - Start REAL trading\n"
            "/stop - Stop execution\n"
            "/reset_demo - Reset paper balance to $10,000\n"
            "/status - Check if running\n"
            "/stats - View live performance\n\n"
            "System trades BTCUSDT perpetual futures at 1H timeframe.\n"
            "Paper trading starts at $10,000 with 2% stop loss.\n"
            "Trades auto-close at 4H horizon or stop loss."
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
    if not MANAGER:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()

    if not user or user.status in {'pending', 'banned'}:
        await update.message.reply_text("⛔ Account not approved." + FOOTER)
        session.close()
        return

    # Build credentials dict based on mode
    try:
        if not user.context:
            user.context = UserContext(telegram_id=user_id)
            session.commit()
        ctx = user.context
        creds = _build_creds_from_context(ctx, live=live)
    except Exception as e:
        if live and "Binance API credentials required for live trading" in str(e):
            logger.warning("User %s tried /start_live without Binance API credentials", user_id)
            await update.message.reply_text(
                "❌ Binance API credentials required for live trading.\n\n"
                "Run: `/setup BINANCE_API_KEY BINANCE_API_SECRET`\n\n"
                "For paper trading without credentials, use `/start_demo`" + FOOTER
            )
            session.close()
            return

        logger.error(f"Credential loading failed for user {user_id}: {e}")
        await update.message.reply_text("❌ Failed to load credentials. Try `/setup` again." + FOOTER)
        session.close()
        return

    session.close()

    mode_str = "LIVE 🔴" if live else "DEMO 🟢"
    notify_signal = _build_signal_notifier(context.bot, user_id)

    try:
        started = await MANAGER.start_session(user_id, creds, on_signal=notify_signal)
        if started:
            # Persist requested mode and active state only after successful start.
            _persist_user_session_flags(user_id, live_mode=live, is_active=True)

            await update.message.reply_text(
                f"🚀 **{mode_str} Trading STARTED**\n\n"
                "✅ Analysis running...\n"
                "👉 **Next Step:** Monitor performance.\n"
                "Run: `/stats`" + FOOTER
            )
        else:
            _persist_user_session_flags(user_id, live_mode=live, is_active=True)
            await update.message.reply_text("⚠️ Engine already running." + FOOTER)
    except Exception as e:
        _persist_user_session_flags(user_id, is_active=False)
        logger.error(f"Failed to start engine for user {user_id}: {e}")
        await update.message.reply_text(f"❌ Failed to start engine: {e}" + FOOTER)

async def start_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=False)

async def start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=True)

async def reset_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    if not MANAGER.is_running(user_id):
        await update.message.reply_text("⚠️ Engine not running. Start with `/start_demo` first." + FOOTER)
        return

    engine = MANAGER.sessions[user_id]
    gen = engine.gen

    if gen.live:
        await update.message.reply_text("❌ Cannot reset a live account. Use `/stop` first." + FOOTER)
        return

    gen.reset_paper_balance()
    await update.message.reply_text(
        f"🔄 **Demo Reset!**\n\n"
        f"💰 Balance: ${gen.paper_balance:,.2f}\n"
        f"All signals and stats cleared.\n"
        f"Engine continues running with fresh state." + FOOTER
    )

async def stop_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    stopped = await MANAGER.stop_session(user_id)
    # Explicit /stop means user opted out of auto-resume.
    _persist_user_session_flags(user_id, is_active=False)

    if stopped:
        await update.message.reply_text(
            "Bzzt. **Engine STOPPED** 🛑\n\n"
            "To resume:\n"
            "`/start_demo` or `/start_live`" + FOOTER
        )
    else:
        await update.message.reply_text("⚠️ Engine not running. Auto-resume disabled." + FOOTER)


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
            if user.context:
                user.context.is_active = False
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
                approve_msg = (
                    "🎉 **Account Approved!**\n\n"
                    "You're all set for paper trading!\n"
                    "Run `/start_demo` to begin (no API keys needed).\n\n"
                    "For live trading later:\n"
                    "`/setup BINANCE_API_KEY BINANCE_API_SECRET`\n"
                    "then `/start_live`"
                )
                await context.bot.send_message(target_id, approve_msg)
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

    # Crypto mode: optional Binance API key + secret (for future live trading)
    # Paper trading works without credentials
    if len(args) < 2:
        await update.message.reply_text(
            "**Crypto Mode Setup**\n\n"
            "For paper trading, no setup needed!\n"
            "Just run `/start_demo`\n\n"
            "For future live trading:\n"
            "`/setup BINANCE_API_KEY BINANCE_API_SECRET`\n\n"
            "**Security Note:** Your keys are encrypted." + FOOTER
        )
        return

    api_key = args[0]
    api_secret = args[1]

    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()
    if user:
        if not user.context:
            user.context = UserContext(telegram_id=user_id)
        user.context.binance_api_key = CRYPTO.encrypt(api_key)
        user.context.binance_api_secret = CRYPTO.encrypt(api_secret)
        session.commit()
        await update.message.reply_text(
            "✅ **Binance Credentials Saved!**\n\n"
            "Run: `/start_demo` or `/start_live`" + FOOTER
        )
    else:
        await update.message.reply_text("⛔ User not found." + FOOTER)
    session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not MANAGER:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    if not MANAGER.is_running(user_id):
        await update.message.reply_text("⚠️ Engine not running." + FOOTER)
        return

    engine = MANAGER.sessions[user_id]
    gen = engine.gen
    mode_label = "LIVE �" if gen.live else "PAPER CRYPTO �"

    wr = gen.get_win_rate_stats()

    recent = gen.signal_log[-5:] if gen.signal_log else []
    if recent:
        trade_lines = []
        for s in recent:
            outcome = s.get('outcome', '')
            tag = ""
            if outcome == "win":
                pnl_usd = s.get('pnl_usd', 0)
                tag = f" W ${pnl_usd:+.0f}" if pnl_usd else " W"
            elif outcome == "loss":
                pnl_usd = s.get('pnl_usd', 0)
                exit_r = " SL" if s.get('exit_reason') == 'stop_loss' else ""
                tag = f" L{exit_r} ${pnl_usd:+.0f}" if pnl_usd else f" L{exit_r}"
            price_fmt = f"{s['close_price']:.2f}"
            trade_lines.append(
                f"- {s['signal']} @ {price_fmt} (P={s['probability']:.3f}){tag}"
            )
        trade_str = "\n".join(trade_lines)
    else:
        trade_str = "(No signals yet)"

    # Balance & PnL
    balance_str = f"${wr.get('paper_balance', 10000):,.2f}"
    pnl_str = wr.get('total_pnl', '$0.00')
    stop_losses = wr.get('stop_losses', 0)

    if wr["evaluated"] > 0:
        wr_section = (
            f"\n**Win Rate**\n"
            f"Win Rate: {wr['win_rate']}% ({wr['wins']}W / {wr['losses']}L)\n"
            f"PnL: {pnl_str}\n"
            f"Evaluated: {wr['evaluated']} | Pending: {wr['pending']}\n"
        )
        if stop_losses > 0:
            wr_section += f"Stop losses: {stop_losses}\n"
    else:
        pending = wr['pending']
        if pending > 0:
            wr_section = f"\n**Win Rate**\n(Evaluating {pending} signal(s)...)\n"
        else:
            wr_section = ""

    instrument = "BTCUSDT 1H"
    msg = (
        f"📊 **{mode_label} Statistics** ({instrument})\n\n"
        f"💰 **Balance:** {balance_str}\n"
        f"📈 **PnL:** {pnl_str}\n"
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
        
    async def post_init(app):
        await _restore_active_sessions(app)

    application = ApplicationBuilder().token(token).post_init(post_init).build()
    
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
    application.add_handler(CommandHandler('reset_demo', reset_demo))
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
