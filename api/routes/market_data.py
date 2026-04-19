import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from src.database.models import User
from src.math_kernel.service import MathKernelService
from src.tasks import simulate_market_data_ingestion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/market", tags=["Market Data"])
math_kernel_service = MathKernelService()

@router.get("/historical", response_model=list[dict[str, Any]])
async def get_historical_data_route(
    symbol: str,
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
):
    """Retrieves simulated historical market data."""
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date format. Use YYYY-MM-DD.")

    return math_kernel_service.get_historical_data(symbol, start_date, end_date)

@router.get("/risk/{portfolio_id}", response_model=dict[str, Any])
async def get_portfolio_risk_metrics_route(
    portfolio_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Retrieves simulated risk metrics for a given portfolio UUID."""
    return math_kernel_service.get_risk_metrics(portfolio_id)

@router.post("/calculations", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_market_calculation(
    calculation_in: dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Calculates total price for a given asset (Noun-based)."""
    symbol = calculation_in.get("symbol")
    quantity = calculation_in.get("quantity")
    price = calculation_in.get("price")

    if not all([symbol, quantity is not None, price is not None]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={"code": "INVALID_PARAMS", "message": "Symbol, quantity, and price are required"}
        )

    try:
        total_price = math_kernel_service.calculate_price(symbol, quantity, price)
        return {
            "symbol": symbol,
            "quantity": quantity,
            "unit_price": price,
            "total_price": round(total_price, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Price calculation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={"code": "CALCULATION_ERROR", "message": "Failed to compute financial metrics"}
        )

@router.post("/ingestions", status_code=status.HTTP_202_ACCEPTED)
async def create_market_ingestion(
    ingestion_in: dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Triggers market data ingestion (Noun-based)."""
    symbol = ingestion_in.get("symbol")
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={"code": "MISSING_SYMBOL", "message": "Market symbol is required for ingestion"}
        )

    simulate_market_data_ingestion.delay(symbol=symbol, num_days=ingestion_in.get("num_days", 30))
    return {
        "id": f"ingest_{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
        "status": "enqueued",
        "symbol": symbol
    }
