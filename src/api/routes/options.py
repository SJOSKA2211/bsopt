"""
Options data routes used by the frontend dashboard.

These endpoints provide a lightweight, in-process mock options chain so that
the UI can render without requiring a live market data feed.
"""

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Query

from src.api.schemas.common import DataResponse

router = APIRouter(prefix="/options", tags=["Options"])


@router.get("/chain")
async def get_options_chain(
    symbol: str = Query("AAPL", description="Underlying symbol"),
    expiry: str = Query("all", description="Expiry bucket filter"),
) -> DataResponse:
    """Return a small synthetic options chain for the requested symbol."""
    symbol = symbol.strip().upper()
    if not symbol.isalnum() or len(symbol) > 10:
        return DataResponse(data=[], message="Invalid symbol format")
        
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
    row_id = 0
    for exp in expiries:
        for strike in strikes:
            moneyness = underlying_price / strike
            row_id += 1
            rows.append(
                {
                    "id": f"{symbol}-{exp}-{strike}",
                    "strike": strike,
                    "expiry": exp,
                    "call_bid": max(1.0, (underlying_price - strike) * 0.4),
                    "call_ask": max(1.2, (underlying_price - strike) * 0.45),
                    "call_last": max(1.1, (underlying_price - strike) * 0.43),
                    "call_volume": 100 * row_id,
                    "call_oi": 500 * row_id,
                    "call_iv": 0.25 + (moneyness - 1.0) * 0.02,
                    "call_delta": 0.5 + (moneyness - 1.0) * 0.4,
                    "call_gamma": 0.02,
                    "put_bid": max(1.0, (strike - underlying_price) * 0.4),
                    "put_ask": max(1.2, (strike - underlying_price) * 0.45),
                    "put_last": max(1.1, (strike - underlying_price) * 0.43),
                    "put_volume": 120 * row_id,
                    "put_oi": 400 * row_id,
                    "put_iv": 0.27 + (1.0 - moneyness) * 0.02,
                    "put_delta": -0.5 + (1.0 - moneyness) * 0.4,
                    "put_gamma": 0.02,
                    "underlying_price": underlying_price,
                }
            )

    return DataResponse(data=rows)
