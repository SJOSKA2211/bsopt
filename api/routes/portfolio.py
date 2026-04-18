from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.crud import create_portfolio as crud_create_portfolio
from src.database.crud import get_portfolio_by_id as crud_get_portfolio_by_id
from src.database.crud import get_portfolios_for_user as crud_get_portfolios_for_user
from src.database.crud import get_user_by_id  # Import CRUD for user lookup
from src.database.crud import update_portfolio as crud_update_portfolio
from src.database.models import User  # Import models
from src.database.session import get_async_db

# Import MathKernelService
from src.math_kernel.service import MathKernelService
from src.schemas.portfolio import Portfolio as PortfolioSchema
from src.schemas.portfolio import PortfolioCreate, PortfolioUpdate  # Import Pydantic schemas
from src.shared.protos import auth_pb2, auth_pb2_grpc

# --- Service Instances ---
math_kernel_service = MathKernelService()

# --- Logging and Configuration ---
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/portfolios", tags=["Portfolios"])

# --- Authentication and Authorization Dependencies ---

async def get_current_user( # Placeholder: Real implementation from api.index.py
    request: Request, db: AsyncSession = Depends(get_async_db), auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client) 
) -> User:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
    parts = auth_header.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    
    token = parts[1]

    try:
        token_validation_response = await auth_client.ValidateToken(auth_pb2.TokenRequest(token=token))
        
        if not token_validation_response.valid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is invalid or expired")
        
        user_id = token_validation_response.user_id
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found in token payload")

        db_user = await get_user_by_id(db, user_id=user_id)
        if not db_user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        
        return db_user

    except grpc.RpcError as e:
        logger.error(f"Auth gRPC error during user retrieval: {e.code()} - {e.details()}")
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.details())
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error during user authentication: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during authentication")

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    return current_user.id

async def check_portfolio_ownership(db: AsyncSession, portfolio_id: str, user_id: str) -> str:
    """Verifies that the portfolio belongs to the specified user."""
    portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=user_id)
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return portfolio.user_id

# --- Portfolio Routes ---

@router.post("/", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_portfolio_item(
    portfolio_in: PortfolioCreate, 
    db: AsyncSession = Depends(get_async_db),
    user_id: str = Depends(get_current_user_id) 
):
    """Creates a new portfolio for the authenticated user."""
    portfolio_data = portfolio_in.dict()
    portfolio_data["user_id"] = user_id
    
    try:
        db_portfolio = await crud_create_portfolio(db, portfolio_data)
        return {
            "id": db_portfolio.id,
            "name": db_portfolio.name,
            "cash": db_portfolio.cash,
            "user_id": db_portfolio.user_id,
            "created_at": db_portfolio.created_at.isoformat() if db_portfolio.created_at else None,
            "updated_at": db_portfolio.updated_at.isoformat() if db_portfolio.updated_at else None,
        }
    except Exception as e:
        logger.error(f"Failed to create portfolio: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create portfolio")

@router.get("/", response_model=list[PortfolioSchema]) 
async def read_portfolios_list(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user), 
    skip: int = 0,
    limit: int = 100
):
    """Retrieves a list of portfolios for the authenticated user."""
    db_portfolios = await crud_get_portfolios_for_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return [PortfolioSchema.from_orm(p) for p in db_portfolios]

@router.get("/{portfolio_id}", response_model=PortfolioSchema) 
async def read_portfolio_item(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) 
):
    """Retrieves a specific portfolio by its ID."""
    db_portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=current_user.id)
    if db_portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return PortfolioSchema.from_orm(db_portfolio)

@router.put("/{portfolio_id}", response_model=PortfolioSchema) 
async def update_portfolio_item(
    portfolio_id: str,
    portfolio_in: PortfolioUpdate, 
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) 
):
    """Updates an existing portfolio."""
    db_portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=current_user.id)
    if db_portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    update_data = portfolio_in.dict(exclude_unset=True) 
    
    try:
        updated_portfolio = await crud_update_portfolio(db, db_portfolio, update_data)
        return PortfolioSchema.from_orm(updated_portfolio)
    except Exception as e:
        logger.error(f"Failed to update portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update portfolio")

# --- Trade Routes ---
# These routes will also require authentication and will use get_current_user dependency.
# They will also need Pydantic schemas for TradeCreate and TradeUpdate.
# (Assuming api/routes/trade.py is updated similarly)

# --- Portfolio Valuation Route ---
@router.get("/value/{portfolio_id}", response_model=dict[str, Any])
async def get_portfolio_value_route(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Calculates and returns the simulated total value of a portfolio.
    """
    # Verify portfolio ownership first
    await check_portfolio_ownership(db, portfolio_id, current_user.id)
    
    try:
        total_value = await math_kernel_service.calculate_portfolio_value(portfolio_id, db)
        return {
            "portfolio_id": portfolio_id,
            "total_value": round(total_value, 2),
            "currency": "USD", # Assuming USD for now
            "calculation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError as ve: # Catch specific error for portfolio not found
        logger.error(f"Error calculating value for portfolio {portfolio_id}: {ve}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Error calculating portfolio value for {portfolio_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate portfolio value")
