from typing import List, Optional
from datetime import datetime


class MarketData:
    def __init__(self, symbol: str, timestamp: datetime, bid: float, ask: float, last: float, volume: int):
        self.symbol = symbol
        self.timestamp = timestamp
        self.bid = bid
        self.ask = ask
        self.last = last
        self.volume = volume

async def get_market_data(
    symbol: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[MarketData]:
    """Dummy resolver for fetching historical market data."""
    print(f"Dummy get_market_data called for {symbol}, {start_time}, {end_time}")
    return [MarketData(symbol=symbol, timestamp=datetime.now(), bid=100.0, ask=100.5, last=100.25, volume=1000)]
