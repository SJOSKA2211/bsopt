import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_current_user_id
from src.database.crud import (
    create_portfolio as crud_create_portfolio,
)
from src.database.crud import (
    get_portfolio_by_id as crud_get_portfolio_by_id,
)
from src.database.crud import (
    get_portfolios_for_user as crud_get_portfolios_for_user,
)
from src.database.models import User
from src.database.session import get_async_db
from src.math_kernel.service import MathKernelService
from src.schemas.portfolio import Portfolio as PortfolioSchema
from src.schemas.portfolio import PortfolioCreate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/portfolios", tags=["Portfolios"])
math_kernel_service = MathKernelService()

async def check_portfolio_ownership(db: AsyncSession, portfolio_id: UUID, user_id: UUID) -> UUID:
    """Verifies that the portfolio belongs to the specified user."""
    portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=user_id)
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return portfolio.user_id

@router.post("/", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_portfolio_item(
    portfolio_in: PortfolioCreate,
    db: AsyncSession = Depends(get_async_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Creates a new portfolio for the authenticated user."""
    portfolio_data = portfolio_in.dict()
    portfolio_data["user_id"] = user_id

    try:
        db_portfolio = await crud_create_portfolio(db, portfolio_data)
        return {
            "id": str(db_portfolio.id),
            "name": db_portfolio.name,
            "cash": db_portfolio.cash,
            "user_id": str(db_portfolio.user_id),
            "created_at": db_portfolio.created_at.isoformat(),
            "updated_at": db_portfolio.updated_at.isoformat(),
        }
    except Exception as e:
        logger.error("Failed to create portfolio: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create portfolio")

@router.get("/", response_model=list[PortfolioSchema])
async def read_portfolios_list(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
):
    """Retrieves a list of portfolios for the authenticated user."""
    db_portfolios = await crud_get_portfolios_for_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return [PortfolioSchema.from_orm(p) for p in db_portfolios]

@router.get("/{portfolio_id}", response_model=PortfolioSchema)
async def read_portfolio_item(
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves a specific portfolio by its UUID."""
    db_portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=current_user.id)
    if db_portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return PortfolioSchema.from_orm(db_portfolio)

@router.get("/value/{portfolio_id}", response_model=dict[str, Any])
async def get_portfolio_value_route(
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Calculates the total value of a portfolio using the math kernel."""
    await check_portfolio_ownership(db, portfolio_id, current_user.id)

    try:
        total_value = await math_kernel_service.calculate_portfolio_value(portfolio_id, db)
        return {
            "portfolio_id": str(portfolio_id),
            "total_value": round(total_value, 2),
            "currency": "USD",
            "calculation_timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error("Error calculating portfolio value for %s: %s", portfolio_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Value calculation failed")
