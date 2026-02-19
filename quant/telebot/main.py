
import logging
import os
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from quant.telebot.models import Base, User, UserContext
from quant.telebot.auth import CryptoManager
from quant.telebot.manager import BotManager

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

# Db
DB_PATH = os.path.abspath("quant_bot.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=ENGINE)

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
                "Your account is **PENDING APPROVAL**.\n"
                "Please contact the administrator."
            )
            
            # Notify Admin
            admin_id = os.getenv("ADMIN_ID")
            if admin_id:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🔔 **New User:** {user.first_name} (@{user.username}) [ID: {user.id}]\n"
                             f"Run `/approve {user.id}` to grant access."
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin: {e}")
        elif db_user.status == 'pending':
            await update.message.reply_text("⏳ Your account is still pending approval.")
        elif db_user.status == 'banned':
            await update.message.reply_text("🚫 Access denied.")
        else:
            await update.message.reply_text(f"✅ Welcome back, {user.first_name}! System is online.")
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("⚠️ System error.")
    finally:
        session.close()

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📚 **Commands:**\n"
        "/start - Register/Check Status\n"
        "/setup <email> <key> <pass> - Set Credentials\n"
        "/mode [demo|live] - Switch Mode\n"
        "/start_trading - Launch Engine\n"
        "/stop - Stop Engine\n"
        "/status - Check Engine Status\n"
    )
    user_id = update.effective_user.id
    admin_id_str = os.getenv("ADMIN_ID", "")
    if str(user_id) == admin_id_str:
        msg += "\n👑 **Admin:**\n/approve <id> - Approve User\n/revoke <id> - Ban User\n"
    
    await update.message.reply_text(msg)

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if str(user_id) != os.getenv("ADMIN_ID", ""):
        return 

    try:
        target_id = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /approve <telegram_id>")
        return

    session = SessionLocal()
    try:
        target_user = session.query(User).filter_by(telegram_id=target_id).first()
        if target_user:
            target_user.status = 'approved'
            session.commit()
            await update.message.reply_text(f"✅ User {target_id} approved!")
            try:
                await context.bot.send_message(target_id, "🎉 **Access Granted!**\nRun `/setup <email> <key> <pass>` to configure.")
            except: pass
        else:
            await update.message.reply_text("❌ User not found.")
    finally:
        session.close()

async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) != 3:
        await update.message.reply_text("Usage: /setup <email> <api_key> <api_password>\n\n⚠️ **Delete your message after sending!**")
        return

    email, key, pw = args[0], args[1], args[2]
    
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(telegram_id=user_id).first()
        if not user or user.status != 'approved':
            await update.message.reply_text("❌ Permission denied.")
            return
            
        # Encrypt
        enc_key = CRYPTO.encrypt(key)
        enc_pass = CRYPTO.encrypt(pw)
        
        ctx = user.context
        if not ctx:
            ctx = UserContext(telegram_id=user_id)
            user.context = ctx
            
        ctx.cap_email = email
        ctx.cap_api_key = enc_key
        ctx.cap_api_pass = enc_pass
        
        session.commit()
        await update.message.reply_text("✅ Credentials saved securely! \nRun `/start_trading` to begin.")
        
        # Try to delete user message
        try:
            await update.message.delete()
        except:
            await update.message.reply_text("⚠️ I could not delete your message. Please delete it yourself for security.")
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        await update.message.reply_text("❌ Setup failed.")
    finally:
        session.close()

async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args or context.args[0].lower() not in ['demo', 'live']:
        await update.message.reply_text("Usage: /mode [demo|live]")
        return
        
    mode = context.args[0].lower()
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(telegram_id=user_id).first()
        if user and user.context:
            user.context.mode = mode
            session.commit()
            await update.message.reply_text(f"🔄 Mode set to **{mode.upper()}**.\n(Restart trading to apply)")
        else:
             await update.message.reply_text("❌ User not found or setup not done.")
    finally:
        session.close()

async def start_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not MANAGER:
        await update.message.reply_text("❌ Bot not initialized (Model not found).")
        return

    if MANAGER.is_running(user_id):
        await update.message.reply_text("⚠️ Trading is already running!")
        return

    session = SessionLocal()
    try:
        user = session.query(User).filter_by(telegram_id=user_id).first()
        if not user or user.status != 'approved':
            await update.message.reply_text("❌ Permission denied.")
            return
            
        ctx = user.context
        if not ctx or not ctx.cap_api_key:
            await update.message.reply_text("❌ No credentials. Run /setup first.")
            return
            
        # Decrypt
        plain_key = CRYPTO.decrypt(ctx.cap_api_key)
        plain_pass = CRYPTO.decrypt(ctx.cap_api_pass)
        
        # Signal Callback
        # Must be robust
        async def on_signal(sig):
            try:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig["signal"], "❓")
                msg = (
                    f"{emoji} **SIGNAL:** {sig['signal']} @ {sig['close_price']}\n"
                    f"📊 Prob: {sig['probability']} (Thresh: {sig['threshold']})\n"
                    f"🧠 Regime: {sig['regime']} ({'✅' if sig['regime_tradeable'] else '❌'})\n"
                    f"📝 Reason: {sig['reason']}"
                )
                if sig['position']:
                    msg += f"\n💰 Size: {sig['position']['lot_size']} lots"
                    
                await context.bot.send_message(user_id, msg)
            except Exception as ex:
                logger.error(f"Failed to send signal to {user_id}: {ex}")

        # Start
        await update.message.reply_text(f"🚀 Starting Engine ({ctx.mode.upper()})...")
        await MANAGER.start_session(
            user_id,
            {
                'email': ctx.cap_email,
                'key': plain_key,
                'password': plain_pass,
                'mode': ctx.mode
            },
            on_signal
        )
        await update.message.reply_text("✅ Engine Running. You will receive alerts here.")
        
    except Exception as e:
        logger.error(f"Start trading failed: {e}")
        await update.message.reply_text(f"❌ Failed to start: {e}")
    finally:
        session.close()

async def stop_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not MANAGER: return

    if await MANAGER.stop_session(user_id):
        await update.message.reply_text("🛑 Trading Engine Stopped.")
    else:
        await update.message.reply_text("⚠️ No active session found.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not MANAGER: 
        await update.message.reply_text("❌ Manager Error.")
        return
        
    running = MANAGER.is_running(user_id)
    state = "Running 🟢" if running else "Stopped 🔴"
    
    await update.message.reply_text(f"🤖 **Bot Status:**\nEngine: {state}")


def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("Error: TELEGRAM_TOKEN environment variable is missing.")
        return
        
    application = ApplicationBuilder().token(token).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('approve', approve))
    application.add_handler(CommandHandler('setup', setup))
    application.add_handler(CommandHandler('mode', set_mode))
    application.add_handler(CommandHandler('start_trading', start_trading))
    application.add_handler(CommandHandler('stop', stop_trading))
    application.add_handler(CommandHandler('status', status))
    
    print("Bot is polling...")
    application.run_polling()

if __name__ == '__main__':
    main()
