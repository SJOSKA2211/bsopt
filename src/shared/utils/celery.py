import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

import structlog
from celery import Task

try:
    import uvloop

    _UVLOOP_AVAILABLE = True
except ImportError:
    _UVLOOP_AVAILABLE = False

logger = structlog.get_logger(__name__)

class BaseAsyncTask(Task):
    """
    High-Performance Celery Task: Manages a persistent process-level event loop.
    Eliminates asyncio.run() overhead and enables high-throughput async delegation.
    """

    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if BaseAsyncTask._loop is None:
            if _UVLOOP_AVAILABLE:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("using_uvloop_for_base_async_task")

            BaseAsyncTask._loop = asyncio.new_event_loop()
            BaseAsyncTask._thread = threading.Thread(
                target=self._run_event_loop, args=(BaseAsyncTask._loop,), daemon=True
            )
            BaseAsyncTask._thread.start()
            logger.info("persistent_event_loop_started", thread_id=BaseAsyncTask._thread.ident)

        return BaseAsyncTask._loop

    def _run_event_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def run_async(self, coro: Coroutine[Any, Any, Any], timeout: float | None = 30.0) -> Any:
        """
        Run a coroutine in the persistent loop and wait for result.
        OPTIMIZED: Uses wait() with timeout to prevent worker thread exhaustion.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error("task_async_timeout", timeout=timeout)
            future.cancel()
            raise

    def run_forget(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Fire-and-forget execution: Dispatches to the loop without blocking.
        Useful for non-critical logging or metrics.
        """
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))
