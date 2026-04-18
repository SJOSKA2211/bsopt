from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.crud import get_user_by_id  # For user lookup in auth dependency

# Import necessary models and services
from src.database.models import User  # Assuming User model is needed for auth context
from src.database.session import get_async_db
from src.math_kernel.service import MathKernelService  # Import the Math Kernel Service
from src.shared.protos import (
    auth_pb2,  # Import proto types
    auth_pb2_grpc,  # Import gRPC stubs
)

# --- Service Instances ---
math_kernel_service = MathKernelService()

# --- Authentication Dependency ---
async def get_current_user( # Placeholder: Real implementation from api.index.py
    request: Request, db: AsyncSession = Depends(get_async_db), auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client),
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
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error during user authentication: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during authentication")

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    return current_user.id

router = APIRouter(prefix="/api/v1/market", tags=["Market Data"])

@router.get("/historical", response_model=list[dict[str, Any]])
async def get_historical_data_route(
    symbol: str,
    start_date: str, # e.g., "2023-01-01"
    end_date: str,   # e.g., "2023-12-31"
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves simulated historical market data for a given symbol and date range.
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

@router.get("/risk/{portfolio_id}", response_model=dict[str, Any])
async def get_portfolio_risk_metrics_route(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves simulated risk metrics for a given portfolio.
    """
    logger.info(f"Fetching simulated risk metrics for portfolio: {portfolio_id}")

    # In a real app, you'd first verify ownership of the portfolio_id

    risk_metrics = math_kernel_service.get_risk_metrics(portfolio_id)

    return risk_metrics

@router.post("/calculate_price", response_model=dict[str, Any])
async def calculate_price_endpoint(
    calculation_data: dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Calculates the total price for a given symbol, quantity, and unit price.
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
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error calculating price: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Price calculation failed")

# --- Endpoint to trigger market data ingestion (using Celery task) ---
@router.post("/ingest_market_data", status_code=status.HTTP_202_ACCEPTED)
async def trigger_market_data_ingestion_route(
    ingestion_params: dict[str, Any], # e.g., {"symbol": "GOOG", "num_days": 30}
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    """Triggers simulated market data ingestion asynchronously using Celery.
    """
    symbol = ingestion_params.get("symbol")
    num_days = ingestion_params.get("num_days", 30)

    if not symbol:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol is required for market data ingestion")

    try:
        simulate_market_data_ingestion.delay(symbol=symbol, num_days=num_days)
        return {
            "message": "Market data ingestion task enqueued successfully",
            "symbol": symbol,
            "num_days": num_days,
            "status": "queued",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to enqueue market data ingestion task for symbol {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enqueue market data ingestion task")

# --- New Endpoint: Get Current Market Prices ---
@router.get("/current_prices", response_model=dict[str, float])
async def get_current_prices_route(
    symbols: list[str], # Query parameter for symbols, e.g., /current_prices?symbols=AAPL&symbols=GOOG
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user), # Ensure user is authenticated
):
    """Retrieves simulated current market prices for a list of symbols.
    """
    if not symbols:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one symbol is required")

    try:
        current_prices = math_kernel_service.get_current_market_prices(symbols)
        return current_prices
    except Exception as e:
        logger.error(f"Error fetching current market prices for symbols {symbols}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch current market prices")

