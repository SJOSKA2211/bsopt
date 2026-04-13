from datetime import datetime

from sqlalchemy import create_engine, text

from api.graphql.types import OHLCV, MarketData
from src.config import settings
from src.ingestion.router import MarketDataRouter
from src.shared.shm_mesh import GreeksMesh

router = MarketDataRouter()
_greeks_mesh = GreeksMesh(create=False)
_engine = create_engine(settings.DATABASE_URL)


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
    Fetch historical data from TimescaleDB (market_ticks).
    """
    query = text("""
        SELECT time as timestamp, price, volume, bid, ask
        FROM market_ticks
        WHERE symbol = :symbol AND time BETWEEN :start AND :end
        ORDER BY time ASC
    """)

    results = []
    with _engine.connect() as conn:
        rows = conn.execute(query, {"symbol": symbol, "start": start_time, "end": end_time})
        for row in rows:
            results.append(
                MarketData(
                    symbol=symbol,
                    timestamp=row.timestamp,
                    bid=row.bid,
                    ask=row.ask,
                    last=row.price,
                    volume=row.volume,
                )
            )

    # Fallback to live point if no history and range is very recent
    if not results and (datetime.now() - start_time).total_seconds() < 3600:
        results.append(await get_market_data(symbol))

    return results


async def get_historical_ohlcv(
    symbol: str, start_time: datetime, end_time: datetime, interval_minutes: int = 1
) -> list[OHLCV]:
    """
    Fetch historical OHLCV data using TimescaleDB time_bucket.
    """
    interval = f"{interval_minutes} minutes"
    query = text("""
        SELECT 
            time_bucket(:interval, time) AS bucket,
            first(price, time) as open,
            max(price) as high,
            min(price) as low,
            last(price, time) as close,
            sum(volume) as volume
        FROM market_ticks
        WHERE symbol = :symbol AND time BETWEEN :start AND :end
        GROUP BY bucket
        ORDER BY bucket ASC
    """)

    results = []
    with _engine.connect() as conn:
        rows = conn.execute(
            query, {"symbol": symbol, "start": start_time, "end": end_time, "interval": interval}
        )
        for row in rows:
            results.append(
                OHLCV(
                    time=row.bucket.isoformat(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=int(row.volume or 0),
                )
            )

    return results