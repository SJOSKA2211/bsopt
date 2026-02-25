"""
Portfolio routes backing the dashboard overview widgets.

These endpoints intentionally return a compact, synthetic snapshot of a
trading portfolio so the frontend can render without a live trading backend.
"""

from fastapi import APIRouter

from src.api.schemas.common import DataResponse

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


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
        "positionsCount": 8,
        "positions": [],
    }
    return DataResponse(data=data)
