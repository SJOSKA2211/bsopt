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
            return MagicMock()

        def agent(self, *args, **kwargs):
            return lambda f: f

    from unittest.mock import MagicMock

from typing import Any


class VolatilityAggregationStream:
    """
    Stream processor for calculating realized volatility in real-time.
    """

    def __init__(self, bootstrap_servers: str = "kafka://localhost:9092"):
        self.app = App("volatility-aggregator", broker=bootstrap_servers)
        self.market_data_topic = self.app.topic("market-data", partitions=16)
        self.volatility_table = self.app.Table(
            "volatility-1min-v2", default=float, partitions=16
        )
        self.price_history = self.app.Table(
            "price-history-v2", default=float, partitions=16
        )

        @self.app.agent(self.market_data_topic)
        async def calculate_realized_volatility_agent(stream):
            async for event in stream:
                await self.calculate_realized_volatility(event)

    async def calculate_realized_volatility(self, event: Any):
        """
        Calculates log-returns and updates the realized variance table.
        """
        symbol = event.get("symbol")
        last_price = event.get("last")

        if not symbol or last_price is None:
            return

        prev_price = self.price_history[symbol]
        if prev_price > 0:
            log_return = np.log(last_price / prev_price)

            # Get current variance (handling window objects if present)
            current_val = self.volatility_table[symbol]
            if hasattr(current_val, "now"):
                current_variance = current_val.now()
            else:
                current_variance = current_val

            self.volatility_table[symbol] = current_variance + log_return**2

        self.price_history[symbol] = last_price
