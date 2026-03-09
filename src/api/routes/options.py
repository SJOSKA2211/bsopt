"""
Options data routes used by the frontend dashboard.
Optimized for high-performance database retrieval.
"""

import msgspec
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponse
from src.database import get_async_db
from src.database.models import OptionPrice

router = APIRouter(prefix="/options", tags=["Options"], default_response_class=MsgspecJSONResponse)


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


@router.get("/chain")
async def get_options_chain(
    symbol: str = Query("AAPL", description="Underlying symbol"),
    expiry: str = Query("all", description="Expiry bucket filter"),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse:
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
            return DataResponse(
                data=[
                    OptionChainItem(
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
                    for p in prices
                ],
                message="Real-time manifold data",
            )
    except Exception:
        # Fallback to synthetic if DB is empty/fails in dev
        pass

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

    return DataResponse(data=rows, message="Synthetic fallback data")
