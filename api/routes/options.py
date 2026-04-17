"""
Options data routes used by the frontend dashboard.
Optimized for high-performance database retrieval.
"""

from datetime import date, timedelta

import msgspec
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.responses import MsgspecJSONResponse
from api.schemas.common import DataResponse
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


@router.get("/greeks/{symbol}", response_model=DataResponse[dict[str, float]])
@db_circuit
async def get_realtime_greeks(
    symbol: str, current_user: User = Depends(get_current_active_user)
) -> DataResponse[dict[str, float]]:
    """Return real-time Greeks from SHM for a symbol."""
    data = _greeks_mesh.read(symbol.upper().strip())
    if not data:
        return DataResponse(data={}, message="No live data in manifold", success=False)
    return DataResponse(data=data)


@router.post("/greeks/batch", response_model=DataResponse[dict[str, dict[str, float]]])
@db_circuit
async def get_batch_greeks(
    symbols: list[str], current_user: User = Depends(get_current_active_user)
) -> DataResponse[dict[str, dict[str, float]]]:
    """Batch lookup of real-time Greeks from SHM."""
    results = {}
    for sym in symbols:
        data = _greeks_mesh.read(sym.upper().strip())
        if data:
            results[sym] = data
    return DataResponse(data=results)


@router.get("/chain", response_model=DataResponse[list[OptionChainItem]])
@db_circuit
async def get_options_chain(
    symbol: str = Query("SPX", description="Underlying symbol"),
    expiry: str = Query("all", description="Expiry bucket filter"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[list[OptionChainItem]]:
    """Return the options chain for the requested symbol (Optimized DB lookup)."""

    symbol = symbol.strip().upper()
    if not symbol.isalnum() or len(symbol) > 10:
        return DataResponse(data=[], message="Invalid symbol format")

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
            enriched_data = []

            # Hoisted exactly O(1) SHM lookup per chain instead of loop bound N calls
            shm_greeks = _greeks_mesh.read(symbol)

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

                if shm_greeks:
                    item.delta = shm_greeks.get("delta", item.delta)
                    item.gamma = shm_greeks.get("gamma", item.gamma)
                    item.vega = shm_greeks.get("vega", item.vega)
                    item.theta = shm_greeks.get("theta", item.theta)
                    item.rho = shm_greeks.get("rho", item.rho)

                enriched_data.append(item)

            return DataResponse(
                data=enriched_data,
                message="Real-time manifold data",
            )
    except Exception as e:
        import structlog

        logger = structlog.get_logger(__name__)
        logger.error("options_chain_db_lookup_failed", error=str(e), symbol=symbol)

    return DataResponse(data=[], message="No option chain data found in persistence layer")
    return DataResponse(data=[], message="No option chain data found in persistence layer")
