import structlog
from faust import App
import numpy as np

logger = structlog.get_logger()

class VolatilityAggregationStream:
    """ Kafka Streams for real-time volatility calculation.
    Aggregates tick data into windowed volatility metrics.
    """
    def __init__(self,
        bootstrap_servers: str = 'kafka://kafka-1:9092'
    ):
        # Create Faust app (Kafka Streams for Python)
        self.app = App(
            'volatility-aggregation',
            broker=bootstrap_servers,
            value_serializer='json',
        )

        # Input stream
        self.market_data_topic = self.app.topic('market-data')

        # Output table (windowed aggregations)
        self.volatility_table = self.app.Table(
            'volatility-1min',
            default=float,
            partitions=8
        )
        self.price_history = {} # Store last price for log return calculation

        # Register agent programmatically after app is defined
        self.app.agent(self.market_data_topic)(self.calculate_realized_volatility)


    async def calculate_realized_volatility(self, stream):
        """ Calculate 1-minute realized volatility from tick data.
        Uses tumbling window aggregation.
        """
        async for event in stream.group_by(lambda x: x['symbol']):
            symbol = event['symbol']
            price = event['last']
            timestamp = event['timestamp']

            # Calculate log return
            if symbol in self.price_history:
                prev_price = self.price_history[symbol]
                log_return = np.log(price / prev_price)

                # Update volatility (running calculation)
                self.volatility_table[symbol] = self._update_volatility(
                    symbol,
                    log_return,
                    timestamp
                )

            # Store price for next calculation
            self.price_history[symbol] = price

    def _update_volatility(self, symbol: str, log_return: float, timestamp: int) -> float:
        """ Update volatility estimate using exponential moving average.
        Annualized volatility = sqrt(252 * 6.5 * 60) * sqrt(E[r²])
        """
        alpha = 0.94  # Decay factor (RiskMetrics)
        current_vol = self.volatility_table.get(symbol, 0.0)

        annualization_factor_sqrt = np.sqrt(252 * 6.5 * 60)
        current_variance_unannualized = (current_vol / annualization_factor_sqrt)**2 if current_vol > 0 else 0.0

        # Update variance estimate (using unannualized variance)
        variance = alpha * current_variance_unannualized + (1 - alpha) * log_return**2

        # Annualize
        annualized_vol = annualization_factor_sqrt * np.sqrt(variance)
        return annualized_vol