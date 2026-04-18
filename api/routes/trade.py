import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.crud import (
    create_trade as crud_create_trade,
    get_portfolio_by_id as crud_get_portfolio_by_id,
    get_trade_by_id as crud_get_trade_by_id,
    get_trades_for_portfolio as crud_get_trades_for_portfolio,
)
from src.database.models import User
from src.database.session import get_async_db
from src.schemas.trade import Trade as TradeSchema
from src.schemas.trade import TradeCreate
from api.index import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/trades", tags=["Trades"])

async def check_portfolio_ownership(db: AsyncSession, portfolio_id: UUID, user_id: UUID) -> UUID:
    """Verifies that the portfolio belongs to the specified user."""
    portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=user_id)
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return portfolio.user_id

@router.post("/", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_trade_item(
    trade_in: TradeCreate,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Creates a new trade for a portfolio owned by the user."""
    # Ensure trade_in.portfolio_id is cast to UUID if it's a string from the schema
    p_id = UUID(str(trade_in.portfolio_id))
    await check_portfolio_ownership(db, p_id, current_user.id)

    trade_data = trade_in.dict()
    trade_data["portfolio_id"] = p_id

    try:
        db_trade = await crud_create_trade(db, trade_data)
        return {
            "id": str(db_trade.id),
            "portfolio_id": str(db_trade.portfolio_id),
            "symbol": db_trade.symbol,
            "quantity": db_trade.quantity,
            "price": db_trade.price,
            "side": db_trade.side,
            "order_type": db_trade.order_type,
            "status": db_trade.status,
            "timestamp": db_trade.created_at.isoformat(),
        }
    except Exception as e:
        logger.error("Failed to create trade: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create trade")

@router.get("/", response_model=list[TradeSchema])
async def read_trades_list(
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
):
    """Retrieves all trades for a specific portfolio."""
    await check_portfolio_ownership(db, portfolio_id, current_user.id)

    db_trades = await crud_get_trades_for_portfolio(db, portfolio_id=portfolio_id, skip=skip, limit=limit)
    return [TradeSchema.from_orm(t) for t in db_trades]

@router.get("/{trade_id}", response_model=TradeSchema)
async def read_trade_item(
    trade_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves a specific trade and verifies ownership."""
    db_trade = await crud_get_trade_by_id(db, trade_id=trade_id)
    if db_trade is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")

    await check_portfolio_ownership(db, db_trade.portfolio_id, current_user.id)
    return TradeSchema.from_orm(db_trade)
