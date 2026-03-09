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

    def run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Run a coroutine in the persistent loop and wait for result.
        Uses run_coroutine_threadsafe for thread-safe cross-loop execution.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
