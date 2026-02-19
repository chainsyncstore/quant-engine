
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
            return

        mode = creds.get('mode', 'demo')
        base_url = "https://api-capital.backend-capital.com" if mode == 'live' else "https://demo-api-capital.backend-capital.com"
        
        logger.info(f"Starting session for user {user_id} in {mode.upper()} mode.")
        
        # Create user-specific config
        api_cfg = CapitalAPIConfig(
            api_key=creds['key'],
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
            
            # Create async engine wrapper
            engine = AsyncEngine(gen, on_signal=on_signal)
            await engine.start()
            self.sessions[user_id] = engine
            logger.info(f"Session started for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start session for user {user_id}: {e}")
            raise e

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
