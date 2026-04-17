from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from datetime import datetime, timezone

from src.database.session import get_async_db
from src.database.crud import (
    create_portfolio as crud_create_portfolio,
    get_portfolio_by_id as crud_get_portfolio_by_id,
    get_portfolios_for_user as crud_get_portfolios_for_user,
    update_portfolio as crud_update_portfolio,
    # Import CRUD for user lookup if needed for portfolio ownership checks
    get_user_by_id 
)
from src.database.models import Portfolio, User # Import models
from src.schemas.portfolio import PortfolioCreate, PortfolioUpdate, Portfolio as PortfolioSchema # Import Pydantic schemas
from src.shared.protos import auth_pb2
from src.shared.protos import auth_pb2_grpc

# Import MathKernelService
from src.math_kernel.service import MathKernelService

# --- Service Instances ---
math_kernel_service = MathKernelService()

# --- Logging and Configuration ---
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/portfolios", tags=["Portfolios"])

# --- Authentication and Authorization Dependencies ---

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client)
) -> User:
    """
    Dependency to get the current authenticated user.
    1. Extracts token from Authorization header.
    2. Calls Auth gRPC service to validate token.
    3. Retrieves user details from the database.
    """
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
    return portfolio.user_id # Return user_id for confirmation if needed

# --- Portfolio Routes ---

@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_portfolio_item(
    portfolio_in: PortfolioCreate, # Use Pydantic schema for request body
    db: AsyncSession = Depends(get_async_db),
    user_id: str = Depends(get_current_user_id) 
):
    """Creates a new portfolio for the authenticated user."""
    
    portfolio_data = portfolio_in.dict()
    portfolio_data["user_id"] = user_id
    
    try:
        db_portfolio = await crud_create_portfolio(db, portfolio_data)
        # Return a simplified response, or the full model if response_model is set appropriately
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

@router.get("/", response_model=List[PortfolioSchema]) # Use schema for response model
async def read_portfolios_list(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user), # Use the authenticated user object
    skip: int = 0,
    limit: int = 100
):
    """Retrieves a list of portfolios for the authenticated user."""
    db_portfolios = await crud_get_portfolios_for_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return [PortfolioSchema.from_orm(p) for p in db_portfolios]

@router.get("/{portfolio_id}", response_model=PortfolioSchema) # Use schema for response model
async def read_portfolio_item(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Use the authenticated user object
):
    """Retrieves a specific portfolio by its ID."""
    db_portfolio = await crud_get_portfolio_by_id(db, portfolio_id=portfolio_id, user_id=current_user.id)
    if db_portfolio is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return PortfolioSchema.from_orm(db_portfolio)

@router.put("/{portfolio_id}", response_model=PortfolioSchema) # Use schema for response model
async def update_portfolio_item(
    portfolio_id: str,
    portfolio_in: PortfolioUpdate, # Use Pydantic schema for request body
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Use the authenticated user object
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
@router.get("/value/{portfolio_id}", response_model=Dict[str, Any])
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
