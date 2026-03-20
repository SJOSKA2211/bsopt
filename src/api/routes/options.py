"""
Options data routes used by the frontend dashboard.
Optimized for high-performance database retrieval.
"""

from datetime import date, timedelta
from typing import Any

import msgspec
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponseStruct
from src.auth.auth import get_current_active_user
from src.database import get_async_db
from src.database.models import OptionPrice, User
from src.shared.shm_mesh import GreeksMesh
from src.shared.utils.circuit_breaker import db_circuit

router = APIRouter(prefix="/options", tags=["Options"], default_response_class=MsgspecJSONResponse)

# Initialize Greeks Mesh reader (Lock-Free)
_greeks_mesh = GreeksMesh(create=False)


class OptionChainItem(msgspec.Struct):
    id: str
    symbol: str
    strike: float
    expiry: str
    option_type: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    iv: float = 0.0
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    time: str | None = None


@router.get("/greeks/{symbol}", response_model=None)
@db_circuit
async def get_realtime_greeks(
    symbol: str, 
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Return real-time Greeks from SHM for a symbol."""
    data = _greeks_mesh.read(symbol.upper().strip())
    if not data:
        return DataResponseStruct(data={}, message="No live data in manifold", success=False)
    return DataResponseStruct(data=data)


@router.post("/greeks/batch", response_model=None)
@db_circuit
async def get_batch_greeks(
    symbols: list[str], 
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Batch lookup of real-time Greeks from SHM."""
    results = {}
    for sym in symbols:
        data = _greeks_mesh.read(sym.upper().strip())
        if data:
            results[sym] = data
    return DataResponseStruct(data=results)


@router.get("/chain", response_model=None)
@db_circuit
async def get_options_chain(
    symbol: str = Query("AAPL", description="Underlying symbol"),
    expiry: str = Query("all", description="Expiry bucket filter"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> Any:
    """Return the options chain for the requested symbol (Optimized DB lookup)."""
    symbol = symbol.strip().upper()
    if not symbol.isalnum() or len(symbol) > 10:
        return DataResponseStruct(data=[], message="Invalid symbol format")

    # 1. Attempt real DB lookup
    try:
        stmt = select(OptionPrice).where(OptionPrice.symbol == symbol)

        # Simple date logic for expiry filters
        today = date.today()
        if expiry == "week":
            stmt = stmt.where(OptionPrice.expiry <= today + timedelta(days=7))
        elif expiry == "month":
            stmt = stmt.where(OptionPrice.expiry <= today + timedelta(days=30))
        elif expiry == "quarter":
            stmt = stmt.where(OptionPrice.expiry <= today + timedelta(days=90))

        stmt = stmt.order_by(OptionPrice.expiry.asc(), OptionPrice.strike.asc())

        result = await db.execute(stmt)
        prices = result.scalars().all()

        if prices:
            # OPTIMIZED: Enrich DB records with real-time SHM greeks
            enriched_data = []
            for p in prices:
                item = OptionChainItem(
                    id=f"{p.symbol}-{p.expiry}-{p.strike}-{p.option_type}",
                    symbol=p.symbol,
                    strike=float(p.strike),
                    expiry=p.expiry.isoformat(),
                    option_type=p.option_type,
                    bid=float(p.bid) if p.bid else 0.0,
                    ask=float(p.ask) if p.ask else 0.0,
                    last=float(p.last) if p.last else 0.0,
                    volume=p.volume or 0,
                    open_interest=p.open_interest or 0,
                    iv=float(p.implied_volatility) if p.implied_volatility else 0.0,
                    delta=float(p.delta) if p.delta is not None else None,
                    gamma=float(p.gamma) if p.gamma is not None else None,
                    vega=float(p.vega) if p.vega is not None else None,
                    theta=float(p.theta) if p.theta is not None else None,
                    rho=float(p.rho) if p.rho is not None else None,
                    time=p.time.isoformat() if p.time else None,
                )

                # Check SHM for live overrides
                shm_greeks = _greeks_mesh.read(p.symbol)
                if shm_greeks:
                    item.delta = shm_greeks["delta"]
                    item.gamma = shm_greeks["gamma"]
                    item.vega = shm_greeks["vega"]
                    item.theta = shm_greeks["theta"]
                    item.rho = shm_greeks["rho"]

                enriched_data.append(item)

            return DataResponseStruct(
                data=enriched_data,
                message="Real-time manifold data",
            )
    except Exception as e:
        import structlog

        logger = structlog.get_logger(__name__)
        logger.error("options_chain_db_lookup_failed", error=str(e), symbol=symbol)
        # We continue to fallback

    # 2. Fallback: Synthetic data generation
    today = date.today()
    expiries: list[str] = []
    if expiry in {"all", "week"}:
        expiries.append((today + timedelta(days=7)).isoformat())
    if expiry in {"all", "month"}:
        expiries.append((today + timedelta(days=30)).isoformat())
    if expiry in {"all", "quarter"}:
        expiries.append((today + timedelta(days=90)).isoformat())
    if not expiries:
        expiries.append((today + timedelta(days=30)).isoformat())

    underlying_price = 120.0
    strikes = [110.0, 115.0, 120.0, 125.0, 130.0]

    rows: list[dict[str, Any]] = []
    for exp in expiries:
        for strike in strikes:
            rows.append(
                {
                    "id": f"{symbol}-{exp}-{strike}-call",
                    "strike": strike,
                    "expiry": exp,
                    "option_type": "call",
                    "bid": max(1.0, (underlying_price - strike) * 0.4),
                    "ask": max(1.2, (underlying_price - strike) * 0.45),
                    "last": max(1.1, (underlying_price - strike) * 0.43),
                    "volume": 100,
                    "iv": 0.25,
                    "delta": 0.5,
                }
            )

    return DataResponseStruct(data=rows, message="Synthetic fallback data")
