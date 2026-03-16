import numpy as np

try:
    from faust import App
except ImportError:
    # Mock for environments without Faust
    class App:
        def __init__(self, *args, **kwargs):
            pass

        def topic(self, *args, **kwargs):
            return MagicMock()

        def Table(self, *args, **kwargs):
            return MagicMock()  # noqa: N802

        def agent(self, *args, **kwargs):
            return lambda f: f

    from unittest.mock import MagicMock

from typing import Any

import structlog

logger = structlog.get_logger()


class VolatilityAggregationStream:
    """
    Stream processor for calculating realized volatility in real-time.
    OPTIMIZED: Uses micro-batching to reduce state-store contention.
    """

    def __init__(self, bootstrap_servers: str = "kafka://localhost:9092"):
        self.app = App("volatility-aggregator", broker=bootstrap_servers)
        self.market_data_topic = self.app.topic("market-data", partitions=16)
        self.volatility_table = self.app.Table("volatility-1min-v2", default=float, partitions=16)
        self.price_history = self.app.Table("price-history-v2", default=float, partitions=16)

        # Micro-batching buffer
        self._buffer: dict[str, float] = {}
        self._batch_size = 50
        self._msg_count = 0

        @self.app.agent(self.market_data_topic)
        async def calculate_realized_volatility_agent(stream):
            async for event in stream:
                await self.calculate_realized_volatility(event)

                self._msg_count += 1
                if self._msg_count >= self._batch_size:
                    await self._flush_buffer()
                    self._msg_count = 0

    async def calculate_realized_volatility(self, event: Any):
        """
        Calculates log-returns and buffers the realized variance.
        """
        symbol = event.get("symbol")
        last_price = event.get("last")

        if not symbol or last_price is None:
            return

        prev_price = self.price_history[symbol]
        if prev_price > 0:
            log_return = np.log(last_price / prev_price)
            # Buffer the square of log return
            self._buffer[symbol] = self._buffer.get(symbol, 0.0) + log_return**2

        self.price_history[symbol] = last_price

    async def _flush_buffer(self):
        """
        Flush aggregated variance to the state table in a single pass.
        """
        for symbol, variance_delta in self._buffer.items():
            current_val = self.volatility_table[symbol]
            self.volatility_table[symbol] = current_val + variance_delta

        self._buffer.clear()
        logger.debug("volatility_buffer_flushed")
