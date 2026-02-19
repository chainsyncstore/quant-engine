
import asyncio
import logging
from quant.live.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

class AsyncEngine:
    def __init__(self, generator: SignalGenerator, on_signal=None):
        self.gen = generator
        self.on_signal = on_signal
        self.running = False
        self.task = None

    async def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        logger.info("Async Engine started.")

    async def stop(self):
        if not self.running:
            return
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Async Engine stopped.")

    async def _loop(self):
        logger.info("Starting async loop for user...")
        while self.running:
            try:
                # Run the blocking run_once in a thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.gen.run_once)
                
                if result and self.on_signal:
                    if asyncio.iscoroutinefunction(self.on_signal):
                        await self.on_signal(result)
                    else:
                        self.on_signal(result)
            except Exception as e:
                logger.error(f"Engine loop error: {e}")
            
            # Sleep 60s
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
