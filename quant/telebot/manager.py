
import asyncio
import logging
from pathlib import Path
from quant.live.signal_generator import SignalGenerator
from quant.config import BinanceAPIConfig, get_research_config
from quant.telebot.engine import AsyncEngine

logger = logging.getLogger(__name__)

class BotManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.sessions = {} # user_id -> AsyncEngine

    async def start_session(self, user_id: int, creds: dict, on_signal):
        """
        Start a trading session for a user.
        creds: {binance_api_key, binance_api_secret, live}
        """
        if user_id in self.sessions:
            logger.info(f"User {user_id} session already active.")
            return False

        live = creds.get('live', False)
        mode_label = "LIVE" if live else "DEMO"
        rcfg = get_research_config()

        logger.info(f"Starting session for user {user_id} in {mode_label} mode ({rcfg.mode}).")

        try:
            if rcfg.mode != "crypto":
                raise RuntimeError(
                    "Legacy FX mode is disabled. Configure MODE=crypto and use Binance credentials."
                )

            # Crypto mode: Binance client
            binance_cfg = None
            if creds.get('binance_api_key') and creds.get('binance_api_secret'):
                base = "https://testnet.binancefuture.com" if not live else "https://fapi.binance.com"
                binance_cfg = BinanceAPIConfig(
                    api_key=creds['binance_api_key'],
                    api_secret=creds['binance_api_secret'],
                    base_url=base,
                )

            # Live mode requires credentials
            if live and not binance_cfg:
                raise RuntimeError(
                    "Binance API credentials required for live trading. "
                    "Run /setup BINANCE_API_KEY BINANCE_API_SECRET first."
                )

            gen = SignalGenerator(
                model_dir=self.model_dir,
                capital=10000.0,
                horizon=4,
                binance_config=binance_cfg,
                live=live,
            )

            if live:
                # Verify credentials and configure account BEFORE starting
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(None, gen.binance_client.authenticate)
                    gen._authenticated = True

                    # Set conservative defaults: 1x leverage, isolated margin
                    symbol = gen.binance_client._cfg.symbol
                    await loop.run_in_executor(
                        None, gen.binance_client.set_leverage, symbol, gen.binance_client._cfg.leverage
                    )
                    await loop.run_in_executor(
                        None, gen.binance_client.set_margin_type, symbol, gen.binance_client._cfg.margin_type
                    )
                    logger.info(f"Binance LIVE ready for user {user_id}")
                except Exception as auth_err:
                    raise RuntimeError(
                        f"Binance authentication failed: {auth_err}. "
                        f"Check your API key and secret with /setup."
                    ) from auth_err
            else:
                # Paper mode: no auth needed for read-only Binance data
                gen._authenticated = True
                logger.info(f"Binance PAPER ready for user {user_id}")

            # Create async engine wrapper
            interval = 3600
            engine = AsyncEngine(gen, on_signal=on_signal, interval=interval)
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
        if user_id not in self.sessions:
            return False
        engine = self.sessions[user_id]
        if not engine.running:
            # Engine died but session wasn't cleaned up
            del self.sessions[user_id]
            return False
        return True

    def get_active_count(self) -> int:
        return len(self.sessions)
