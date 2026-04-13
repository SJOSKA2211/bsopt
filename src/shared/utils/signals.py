import asyncio
import signal
from collections.abc import Callable
from types import FrameType
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class GracefulShutdown:
    """
    Utility to handle SIGTERM and SIGINT for graceful service shutdown.
    """

    def __init__(self):
        self._callbacks: list[Callable[[], Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None

    def register_callback(self, callback: Callable[[], Any]):
        """Register a cleanup function to be called on shutdown."""
        self._callbacks.append(callback)

    def _handle_signal(self, sig: int, frame: FrameType | None = None):
        sig_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        logger.info("shutdown_signal_received", signal=sig_name)

        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Run callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback) and self._loop is not None:
                    self._loop.create_task(callback())
                else:
                    callback()
            except Exception as e:
                logger.error("shutdown_callback_failed", error=str(e))

    def setup(self, loop: asyncio.AbstractEventLoop | None = None):
        """Setup signal handlers."""
        self._loop = loop or asyncio.get_event_loop()
        self._shutdown_event = asyncio.Event()

        if self._loop is not None:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._loop.add_signal_handler(sig, lambda s=sig: self._handle_signal(s))
                except (NotImplementedError, AttributeError):
                    # Fallback for Windows or systems without loop.add_signal_handler
                    signal.signal(sig, self._handle_signal)

    async def wait_for_shutdown(self):
        """Wait until a shutdown signal is received."""
        if self._shutdown_event is None:
            self.setup()

        if self._shutdown_event is not None:
            await self._shutdown_event.wait()
        logger.info("initiating_final_shutdown")


shutdown_handler = GracefulShutdown()