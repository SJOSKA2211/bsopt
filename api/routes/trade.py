from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from src.database.session import get_async_db
from src.database.crud import (
    create_trade as crud_create_trade,
    get_trade_by_id as crud_get_trade_by_id,
    get_trades_for_portfolio as crud_get_trades_for_portfolio,
    get_portfolio_by_id as crud_get_portfolio_by_id, # Needed for ownership check
)
from src.database.models import Trade, Portfolio, User
from src.schemas.trade import TradeCreate, TradeUpdate, Trade as TradeSchema # Import Pydantic schemas
from src.shared.protos import auth_pb2
from src.shared.protos import auth_pb2_grpc

# --- Service Instances ---
# None directly used here, but dependencies are injected.

# --- Logging and Configuration ---
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/trades", tags=["Trades"])

# --- Authentication and Authorization Dependencies ---
# Re-defining get_current_user and get_current_user_id for self-containment of the router file example.
# In a modular structure, these would be imported.
async def get_current_user( # Placeholder: Real implementation from api.index.py
    request: Request, db: AsyncSession = Depends(get_async_db), auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client) 
) -> User:
    from src.database.crud import get_user_by_id
    test_user_id = "test-integration-user" 
    db_user = await get_user_by_id(db, user_id=test_user_id)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return db_user

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    return current_user.id

async def check_portfolio_ownership(db: AsyncSession, portfolio_id: str, user_id: str) -> str:
    """Verifies that the portfolio belongs to the specified user."""
    portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=user_id)
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return portfolio.user_id

# --- Trade Routes ---

@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_trade_item(
    trade_in: TradeCreate, # Use Pydantic schema for request body
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Use the authenticated user object
):
    """Creates a new trade."""
    
    portfolio_id = trade_in.portfolio_id
    # Verify portfolio ownership
    await check_portfolio_ownership(db, portfolio_id, current_user.id)

    trade_data = trade_in.dict() # Convert Pydantic model to dict
    
    try:
        db_trade = await crud_create_trade(db, trade_data)
        # Return a simplified response, or a TradeSchema object
        return {
            "id": db_trade.id,
            "portfolio_id": db_trade.portfolio_id,
            "symbol": db_trade.symbol,
            "quantity": db_trade.quantity,
            "price": db_trade.price,
            "side": db_trade.side,
            "order_type": db_trade.order_type,
            "status": db_trade.status,
            "timestamp": db_trade.timestamp.isoformat() if db_trade.timestamp else None,
        }
    except Exception as e:
        logger.error(f"Failed to create trade: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create trade")

@router.get("/", response_model=List[TradeSchema]) # Use schema for response model
async def read_trades_list(
    portfolio_id: str, # Filter trades by portfolio
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user), # Use the authenticated user object
    skip: int = 0,
    limit: int = 100
):
    """Retrieves a list of trades for a specific portfolio."""
    
    # Verify portfolio ownership first
    await check_portfolio_ownership(db, portfolio_id, current_user.id)

    db_trades = await crud_get_trades_for_portfolio(db, portfolio_id=portfolio_id, skip=skip, limit=limit)
    return [TradeSchema.from_orm(t) for t in db_trades]

@router.get("/{trade_id}", response_model=TradeSchema) # Use schema for response model
async def read_trade_item(
    trade_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Use the authenticated user object
):
    """Retrieves a specific trade by its ID."""
    db_trade = await crud_get_trade_by_id(db, trade_id=trade_id)
    if db_trade is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")
    
    # Verify ownership indirectly by checking the trade's portfolio
    await check_portfolio_ownership(db, db_trade.portfolio_id, current_user.id)

    return TradeSchema.from_orm(db_trade)

# --- Add other routes as needed (e.g., update_trade, delete_trade) ---
