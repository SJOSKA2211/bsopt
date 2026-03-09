from datetime import datetime

import strawberry

from src.data.router import MarketDataRouter

router = MarketDataRouter()


@strawberry.type
class MarketData:
    symbol: str
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: int | None = None


async def get_market_data(symbol: str) -> MarketData:
    """Fetch live market data for a symbol (Adaptive Routing)."""
    data = await router.get_live_quote(symbol)

    return MarketData(
        symbol=symbol,
        timestamp=datetime.now(),
        bid=data.get("bid"),
        ask=data.get("ask"),
        last=data.get("price"),
        volume=data.get("volume"),
    )


async def get_historical_data(
    symbol: str, start_time: datetime, end_time: datetime
) -> list[MarketData]:
    """
    Fetch historical data (Placeholder - requires TimescaleDB integration).
    """
    # For now, return a single point using the router
    data = await router.get_live_quote(symbol)
    return [
        MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=data.get("bid"),
            ask=data.get("ask"),
            last=data.get("price"),
            volume=data.get("volume"),
        )
    ]
