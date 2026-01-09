import faust
import numpy as np
import os
from typing import Dict
import structlog

logger = structlog.get_logger()

class VolatilityAggregationStream:
    """
    Kafka Streams for real-time volatility calculation.
    Aggregates tick data into windowed volatility metrics using Faust.
    """
    def __init__(self, bootstrap_servers: str = "kafka://localhost:9092"):
        self.app = faust.App(
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
        
        self.price_history = {}
        
        # Define the agent
        self.app.agent(self.market_data_topic)(self.calculate_realized_volatility)

    async def calculate_realized_volatility(self, stream):
        """
        Calculate 1-minute realized volatility from tick data.
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
                
                logger.debug(
                    "volatility_updated", 
                    symbol=symbol, 
                    vol=self.volatility_table[symbol]
                )
                
            # Store price for next calculation
            self.price_history[symbol] = price

    def _update_volatility(
        self, 
        symbol: str, 
        log_return: float, 
        timestamp: int
    ) -> float:
        """
        Update volatility estimate using exponential moving average.
        Annualized volatility = sqrt(252 * 6.5 * 60) * sqrt(E[r²])
        """
        alpha = 0.94  # Decay factor (RiskMetrics)
        current_vol = self.volatility_table.get(symbol, 0.0)
        
        # Update variance estimate (simplified for streaming)
        # Annualized scaling factor for 1-min data: sqrt(252 trading days * 6.5 hours * 60 minutes)
        scaling_factor = np.sqrt(252 * 6.5 * 60)
        
        # Convert annualized vol back to per-minute standard deviation
        current_std = current_vol / scaling_factor
        
        # Update variance
        new_variance = alpha * (current_std**2) + (1 - alpha) * (log_return**2)
        
        # New annualized vol
        new_annualized_vol = np.sqrt(new_variance) * scaling_factor
        
        return new_annualized_vol

    def start(self):
        """Start the Faust application"""
        self.app.main()
