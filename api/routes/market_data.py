from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from src.database.session import get_async_db
# Import necessary models and services
from src.database.models import User # Assuming User model is needed for auth context
from src.math_kernel.service import MathKernelService # Import the Math Kernel Service
from src.database.crud import get_user_by_id # For user lookup in auth dependency

# --- Service Instances ---
math_kernel_service = MathKernelService()

# --- Authentication Dependency ---
# Re-defining get_current_user here for self-containment of the router file example.
# In a modular structure, it would be imported from api.index.
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
    
    # Basic validation for date format (could be more robust with Pydantic)
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date format. Use YYYY-MM-DD.")

    # Use MathKernelService to simulate fetching data
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

