from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from src.database.session import get_async_db
from src.database.crud import (
    # Import CRUD operations if needed for validation, e.g., checking symbol existence
    # get_symbol_details
)
from src.database.models import User # Assuming User model is needed for auth context
from src.math_kernel.service import MathKernelService # Import the Math Kernel Service
from src.tasks import simulate_market_data_ingestion # Import Celery task

# --- Service Instances ---
math_kernel_service = MathKernelService()

# --- Authentication Dependency ---
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

router = APIRouter(prefix="/api/v1/market", tags=["Market Data"])

@router.get("/historical", response_model=List[Dict[str, Any]])
async def get_historical_data(
    symbol: str,
    start_date: str, # e.g., "2023-01-01"
    end_date: str,   # e.g., "2023-12-31"
    db: AsyncSession = Depends(get_async_db), # DB session might be needed for validation or config
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Retrieves simulated historical market data for a given symbol and date range.
    """
    logger.info(f"Request for historical data: Symbol={symbol}, Start={start_date}, End={end_date}")
    
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date format. Use YYYY-MM-DD.")

    historical_data = math_kernel_service.get_historical_data(symbol, start_date, end_date)
    
    if not historical_data:
        logger.warning(f"No historical data simulated for symbol {symbol} in the specified range.")
        return [] 

    return historical_data

@router.get("/risk/{portfolio_id}", response_model=Dict[str, Any])
async def get_portfolio_risk_metrics(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Retrieves simulated risk metrics for a given portfolio.
    """
    logger.info(f"Fetching simulated risk metrics for portfolio: {portfolio_id}")
    
    # In a real app, you'd first verify ownership of the portfolio_id
    # For now, directly call the service.
    
    risk_metrics = math_kernel_service.get_risk_metrics(portfolio_id)
    
    return risk_metrics

# --- New Endpoint: Calculate Price ---
@router.post("/calculate_price", response_model=Dict[str, Any])
async def calculate_price_endpoint(
    calculation_data: Dict[str, Any], # e.g., {"symbol": "AAPL", "quantity": 10, "price": 150.50}
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Calculates the total price for a given symbol, quantity, and unit price.
    """
    symbol = calculation_data.get("symbol")
    quantity = calculation_data.get("quantity")
    price = calculation_data.get("price")

    if not all([symbol, quantity is not None, price is not None]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing required parameters: symbol, quantity, price")
    
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Quantity must be a positive number")
    if not isinstance(price, (int, float)) or price <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Price must be a positive number")

    try:
        total_price = math_kernel_service.calculate_price(symbol, quantity, price)
        return {
            "symbol": symbol,
            "quantity": quantity,
            "unit_price": price,
            "total_price": round(total_price, 2),
            "calculation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating price: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Price calculation failed")

# --- New Endpoint: Trigger Market Data Ingestion Task ---
@router.post("/ingest_market_data", status_code=status.HTTP_202_ACCEPTED)
async def trigger_market_data_ingestion(
    ingestion_params: Dict[str, Any], # e.g., {"symbol": "GOOG", "num_days": 30}
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Triggers simulated market data ingestion asynchronously using Celery.
    """
    symbol = ingestion_params.get("symbol")
    num_days = ingestion_params.get("num_days", 30) # Default to 30 days if not provided

    if not symbol:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol is required for market data ingestion")

    try:
        simulate_market_data_ingestion.delay(symbol=symbol, num_days=num_days)
        return {"message": "Market data ingestion task enqueued successfully", "symbol": symbol, "num_days": num_days, "status": "accepted"}
    except Exception as e:
        logger.error(f"Failed to enqueue market data ingestion task for symbol {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enqueue market data ingestion task")

