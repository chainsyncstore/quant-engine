
import asyncio
import logging
from pathlib import Path
from quant.live.signal_generator import SignalGenerator
from quant.config import CapitalAPIConfig
from quant.telebot.engine import AsyncEngine

logger = logging.getLogger(__name__)

class BotManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.sessions = {} # user_id -> AsyncEngine

    async def start_session(self, user_id: int, creds: dict, on_signal):
        """
        Start a trading session for a user.
        creds: {email, key, password, mode='demo'|'live'}
        """
        if user_id in self.sessions:
            logger.info(f"User {user_id} session already active.")
            return False

        live = creds.get('live', False)
        base_url = "https://api-capital.backend-capital.com" if live else "https://demo-api-capital.backend-capital.com"
        mode_label = "LIVE" if live else "DEMO"

        logger.info(f"Starting session for user {user_id} in {mode_label} mode.")

        # Create user-specific config
        api_cfg = CapitalAPIConfig(
            api_key=creds['api_key'],
            password=creds['password'],
            identifier=creds['email'],
            base_url=base_url
        )
        
        try:
            # Initialize generator with user creds
            gen = SignalGenerator(
                model_dir=self.model_dir,
                capital=10000.0, # TODO: User configurable capital
                horizon=10,
                api_config=api_cfg
            )

            # Test authentication BEFORE starting the loop
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, gen.client.authenticate)
                gen._authenticated = True
                logger.info(f"Auth OK for user {user_id} on {mode_label}")
            except Exception as auth_err:
                error_body = ""
                if hasattr(auth_err, 'response') and auth_err.response is not None:
                    try:
                        error_body = auth_err.response.json().get('errorCode', '')
                    except Exception:
                        error_body = auth_err.response.text[:200]

                if "null.accountId" in str(error_body):
                    hint = (
                        f"Your API key works on LIVE but not on DEMO. "
                        f"Capital.com demo and live are separate environments. "
                        f"Either create an API key in your demo account "
                        f"(log in at demo-trading.capital.com) or use /start_live instead."
                    )
                    raise RuntimeError(hint) from auth_err
                else:
                    raise RuntimeError(
                        f"Authentication failed on {mode_label}: {auth_err}. "
                        f"Check your credentials with /setup."
                    ) from auth_err

            # Create async engine wrapper
            engine = AsyncEngine(gen, on_signal=on_signal)
            await engine.start()
            self.sessions[user_id] = engine
            logger.info(f"Session started for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start session for user {user_id}: {e}")
            raise

    async def stop_session(self, user_id: int):
        if user_id in self.sessions:
            logger.info(f"Stopping session for user {user_id}...")
            engine = self.sessions[user_id]
            await engine.stop()
            del self.sessions[user_id]
            logger.info(f"Session stopped for user {user_id}")
            return True
        return False

    def is_running(self, user_id: int) -> bool:
        return user_id in self.sessions
    
    def get_active_count(self) -> int:
        return len(self.sessions)
