"""FastAPI application setup and configuration."""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Import routers from their respective files
from api.routes.market_data import router as market_data_router
from api.routes.ml import router as ml_router
from api.routes.portfolio import router as portfolio_router
from api.routes.trade import router as trade_router

# Import authentication and database session dependencies
from src.database.session import get_async_db
from src.config import settings # Assuming settings are loaded from .env or similar
from src.database.models import User # For type hinting

# --- Constants ---
# CORS origins: Adjust based on your frontend's domain
DEFAULT_ORIGINS = ["*"] # Allow all origins for development, restrict in production

# --- Logging ---
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="BSOPT API",
    description="Backend API for the Black-Scholes Advanced Option Pricing Platform",
    version="2.5.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# --- Middlewares ---
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add other middlewares here, e.g., rate limiting, authentication

# --- Dependencies ---
# These would typically be defined in a common dependencies file or imported
# For example, get_current_user and get_async_db are used in routers

async def get_current_user(request: Request, db: AsyncSession = Depends(get_async_db)) -> User:
    """
    Placeholder dependency to retrieve the current authenticated user.
    In a real application, this would validate tokens and fetch user data.
    """
    # This is a simplified mock for demonstration. Replace with actual auth logic.
    # For integration tests, you might use a specific user ID or token.
    user_id = request.headers.get("X-User-ID", "mock-user-id-123") # Example: Get user ID from header
    if user_id == "mock-user-id-123":
        # Simulate fetching user from DB
        # For actual DB interaction, use get_user_by_id from crud
        from src.database.crud import get_user_by_id # Avoid circular imports if possible
        db_user = await get_user_by_id(db, user_id=user_id)
        if not db_user:
            # Simulate user not found
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return db_user
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user identifier")

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    """Dependency to retrieve the current user's ID."""
    return current_user.id

# --- Routers ---
# Include routers from their respective files
app.include_router(market_data_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(portfolio_router, prefix="/api/v1")
app.include_router(trade_router, prefix="/api/v1")
# Example: app.include_router(auth_router, prefix="/api/v1/auth")


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        A dictionary indicating the API's status.
    """
    return {"status": "ok", "message": "API is healthy"}


# --- Root Endpoint ---
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint for the API."""
    return {"message": "Welcome to the BSOPT API!"}


# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> HTTPException:
    """Custom handler for FastAPI's HTTPException."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.method} {request.url}")
    return exc # Re-raise the HTTPException to be handled by FastAPI

@app.exception_handler(grpc.RpcError)
async def grpc_exception_handler(request: Request, exc: grpc.RpcError) -> HTTPException:
    """Custom handler for gRPC RpcErrors."""
    logger.error(f"gRPC RpcError: Code={exc.code()}, Details={exc.details()}")
    # Map gRPC status codes to appropriate HTTP status codes
    if exc.code() == grpc.StatusCode.UNAUTHENTICATED:
        status_code = status.HTTP_401_UNAUTHORIZED
    elif exc.code() == grpc.StatusCode.PERMISSION_DENIED:
        status_code = status.HTTP_403_FORBIDDEN
    elif exc.code() == grpc.StatusCode.NOT_FOUND:
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code() == grpc.StatusCode.UNAVAILABLE:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    detail = exc.details() if exc.details() else "gRPC service error"
    return HTTPException(status_code=status_code, detail=detail)

# Consider adding handlers for other common exceptions like ValidationError from Pydantic, etc.
st: The incoming FastAPI request.
        db: SQLAlchemy asynchronous database session.
        auth_client: The gRPC client stub for the Auth service.

    Returns:
        User: The authenticated user object.

    Raises:
        HTTPException: If authentication fails (missing header, invalid token, user not found, or service errors).
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
    except HTTPException:
        raise # Re-raise HTTPExceptions raised within the try block
    except Exception as e:
        logger.exception(f"Unexpected error during user authentication: {e}") # Use logger.exception for traceback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during authentication")


async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    """
    Dependency to retrieve the current user's ID.
    Relies on get_current_user to authenticate and fetch the user.
    """
    return current_user.id

# --- Application Setup ---
app = FastAPI(
    title="BSOPT API",
    description="Backend API for the Black-Scholes Advanced Option Pricing Platform",
    version="2.5.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# --- Middlewares ---
# CORS Middleware for allowing cross-origin requests
origins = ["*"] # Adjust based on security requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (assuming it's defined and added here)
# app.middleware("http")(rate_limit_middleware) # Uncomment when rate_limit_middleware is added to the app


# --- Routers ---
# Import routers from their respective files
from api.routes.market_data import router as market_data_router
from api.routes.ml import router as ml_router
from api.routes.portfolio import router as portfolio_router
from api.routes.trade import router as trade_router
# from api.routes.auth import router as auth_router # Assuming auth routes are handled separately or imported here


# Include routers with API version prefix
app.include_router(market_data_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(portfolio_router, prefix="/api/v1")
app.include_router(trade_router, prefix="/api/v1")
# app.include_router(auth_router, prefix="/api/v1/auth") # Example for auth router


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "API is healthy"}


# --- Root Endpoint ---
@app.get("/")
async def root():
    """Root endpoint for API."""
    return {"message": "Welcome to the BSOPT API!"}


# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTPExceptions."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.method} {request.url}")
    return HTTPException(status_code=exc.status_code, detail=exc.detail)

@app.exception_handler(grpc.RpcError)
async def grpc_exception_handler(request: Request, exc: grpc.RpcError):
    """Custom handler for gRPC RpcErrors."""
    logger.error(f"gRPC RpcError: Code={exc.code()}, Details={exc.details()}")
    # Map gRPC status codes to appropriate HTTP status codes
    if exc.code() == grpc.StatusCode.UNAUTHENTICATED:
        status_code = status.HTTP_401_UNAUTHORIZED
    elif exc.code() == grpc.StatusCode.PERMISSION_DENIED:
        status_code = status.HTTP_403_FORBIDDEN
    elif exc.code() == grpc.StatusCode.NOT_FOUND:
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code() == grpc.StatusCode.UNAVAILABLE:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    detail = exc.details() if exc.details() else "gRPC service error"
    return HTTPException(status_code=status_code, detail=detail)

# Consider adding handlers for other common exceptions like ValidationError from Pydantic, etc.


# --- Helper for testing authentication dependencies ---
# This is a placeholder for demonstration and testing purposes.
# In a real application, this would be part of your authentication setup.
class MockAuthServiceStub:
    async def ValidateToken(self, request: auth_pb2.TokenRequest) -> auth_pb2.ValidateTokenResponse:
        if request.token == "valid-token-abc":
            return auth_pb2.ValidateTokenResponse(valid=True, user_id="test-user-123", email="test@example.com", tier="premium", full_name="User Full Name Placeholder", roles=["user"])
        elif request.token == "revoked-token-xyz":
            return auth_pb2.ValidateTokenResponse(valid=False, user_id="test-user-abc", email="revoked@example.com", tier="free", full_name="Revoked User", roles=["user"], error_message="Token has been revoked")
        else:
            return auth_pb2.ValidateTokenResponse(valid=False, error_message="Token is invalid or expired")

    async def GetUserInfo(self, request: auth_pb2.GetUserInfoRequest) -> auth_pb2.GetUserInfoResponse:
        if request.token == "valid-token-abc":
            return auth_pb2.GetUserInfoResponse(user_id="test-user-123", email="test@example.com", tier="premium", full_name="User Full Name Placeholder", roles=["user"], is_verified=True)
        elif request.token == "token-for-verified-user":
            return auth_pb2.GetUserInfoResponse(user_id="verified-user-id", email="verified@example.com", tier="free", full_name="Verified User", roles=["user"], is_verified=True)
        else:
            raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

    async def RevokeToken(self, request: auth_pb2.RevokeTokenRequest) -> auth_pb2.RevokeTokenResponse:
        # Mock implementation: In reality, this would invalidate the token
        # For testing purposes, we might add it to a revoked list
        print(f"Token {request.token} revoked.")
        return auth_pb2.RevokeTokenResponse() # Empty message

    async def ValidateAPIKey(self, request: auth_pb2.APIKeyRequest) -> auth_pb2.ValidateAPIKeyResponse:
        raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

    async def IntrospectToken(self, request: auth_pb2.IntrospectTokenRequest) -> auth_pb2.IntrospectTokenResponse:
        raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

# Mock dependency for get_auth_client during testing
async def mock_get_auth_client():
    yield MockAuthServiceStub()

# Override the real dependency with the mock for testing auth routes if needed
# For example, in tests/integration/auth/test_auth_grpc.py, the auth_service_servicer fixture likely does this.
# Replace the actual get_auth_client with mock_get_auth_client if needed for specific test setups.

# Example of modifying dependencies for testing (if necessary and possible within this context)
# For demonstration, we'll assume this can be done, but it depends on FastAPI's dependency overrides.
# In tests, you'd typically use app.dependency_overrides.
# For example:
# app.dependency_overrides[get_auth_client] = mock_get_auth_client


# --- Testing Helper ---
# This section is primarily for testing purposes and might be refactored or removed in production.
# If this file is intended to be a runnable application (e.g., with uvicorn),
# ensure it's properly guarded or structured.
# The existence of `uvicorn[standard]` and `granian` in dependencies suggests it is.

# Example for testing purposes - normally, you would run this with uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
