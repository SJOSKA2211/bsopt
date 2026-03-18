from datetime import datetime

import strawberry

from src.ingestion.router import MarketDataRouter
from src.shared.shm_mesh import GreeksMesh

router = MarketDataRouter()
_greeks_mesh = GreeksMesh(create=False)


@strawberry.federation.type(shareable=True)
class MarketData:
    symbol: str
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: int | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


async def get_market_data(symbol: str) -> MarketData:
    """Fetch live market data for a symbol (Adaptive Routing)."""
    data = await router.get_live_quote(symbol)

    md = MarketData(
        symbol=symbol,
        timestamp=datetime.now(),
        bid=data.get("bid"),
        ask=data.get("ask"),
        last=data.get("price"),
        volume=data.get("volume"),
    )

    # Enrich with real-time SHM Greeks
    shm_greeks = _greeks_mesh.read(symbol)
    if shm_greeks:
        md.delta = shm_greeks["delta"]
        md.gamma = shm_greeks["gamma"]
        md.theta = shm_greeks["theta"]
        md.vega = shm_greeks["vega"]
        md.rho = shm_greeks["rho"]

    return md


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
