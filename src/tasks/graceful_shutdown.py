import asyncio
import logging
import signal
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

class ShutdownManager:
    """
    Manages graceful shutdown of the application by handling signals
    and executing registered cleanup tasks.
    """
    def __init__(self):
        self.cleanup_tasks: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._shutdown_event = asyncio.Event()

    def register_cleanup(self, task: Callable[[], Coroutine[Any, Any, None]]):
        """Register a coroutine function to be called during shutdown."""
        self.cleanup_tasks.append(task)

    async def shutdown(self, sig=None):
        """Execute all cleanup tasks."""
        if sig:
            logger.info(f"Received exit signal {sig.name}...")
        
        logger.info("Starting graceful shutdown...")
        
        for task_func in self.cleanup_tasks:
            try:
                logger.info(f"Executing cleanup task: {task_func.__name__}")
                await task_func()
            except Exception as e:
                logger.error(f"Error in cleanup task {task_func.__name__}: {e}")
        
        logger.info("Graceful shutdown complete.")
        
        # Stop the loop
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        
        if tasks:
            logger.info(f"Cancelling {len(tasks)} outstanding tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
            
        loop = asyncio.get_event_loop()
        loop.stop()

    def install_signal_handlers(self):
        """Install signal handlers for SIGINT and SIGTERM."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

shutdown_manager = ShutdownManager()
