"""
Portfolio routes backing the dashboard overview widgets.

These endpoints intentionally return a compact, synthetic snapshot of a
trading portfolio so the frontend can render without a live trading backend.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas.common import DataResponse

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

# In-memory position store (resets on restart – for testing only)
_positions: dict[str, dict] = {}


@router.get("")
@router.get("/")
async def get_portfolio() -> dict:
    """Return the user's portfolio overview including positions."""
    return {
        "balance": 150_000.00,
        "frozen_capital": 25_000.00,
        "risk_score": 0.35,
        "totalValue": 175_000.00,
        "dailyPnL": 1_250.75,
        "dailyPnLPercent": 0.72,
        "positionsCount": len(_positions),
        "positions": list(_positions.values()),
    }


@router.get("/summary")
async def get_portfolio_summary() -> DataResponse:
    """Return a simple, static portfolio summary used by the UI."""
    data = {
        "balance": 150_000.00,
        "frozen_capital": 25_000.00,
        "risk_score": 0.35,
        "totalValue": 175_000.00,
        "dailyPnL": 1_250.75,
        "dailyPnLPercent": 0.72,
        "positionsCount": len(_positions),
        "positions": list(_positions.values()),
    }
    return DataResponse(data=data)


@router.post("/positions", status_code=201)
async def add_position(payload: dict[str, Any]) -> dict:
    """Add a new position to the in-memory portfolio store."""
    # Validate required fields
    symbol = payload.get("symbol")
    quantity = payload.get("quantity")
    entry_price = payload.get("entry_price")

    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=422, detail="'symbol' is required and must be a string")
    if quantity is None or not isinstance(quantity, int | float) or quantity <= 0:
        raise HTTPException(status_code=422, detail="'quantity' must be a positive number")
    if entry_price is None or not isinstance(entry_price, int | float) or entry_price <= 0:
        raise HTTPException(status_code=422, detail="'entry_price' must be a positive number")

    position_id = str(uuid.uuid4())
    position = {
        "id": position_id,
        "symbol": symbol.upper().strip(),
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": float(entry_price),
        "pnl": 0.0,
    }
    _positions[position_id] = position
    return position


@router.delete("/positions/{position_id}")
async def delete_position(position_id: str) -> dict:
    """Delete a position by ID."""
    if position_id not in _positions:
        raise HTTPException(status_code=404, detail="Position not found")
    del _positions[position_id]
    return {"message": "Position deleted", "id": position_id}
