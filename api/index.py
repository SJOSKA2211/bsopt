import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
import grpc
from src.shared.protos import auth_pb2
from src.shared.protos import auth_pb2_grpc
import os
from datetime import datetime, timezone, timedelta

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BS-OPT API", version="6.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth Service Client ---
AUTH_SVC_ADDR = os.getenv("AUTH_SVC_ADDR", "auth_service:50051")
GRPC_CLIENT_CERT_PATH = "/etc/ssl/certs/api_service.crt"
GRPC_CLIENT_KEY_PATH = "/etc/ssl/private/api_service.key"
GRPC_CA_CERT_PATH = "/etc/ssl/certs/root_ca.crt"

async def get_auth_client():
    """Provides a secure gRPC channel to the Auth service using mTLS."""
    channel = None
    try:
        with open(GRPC_CLIENT_CERT_PATH, 'rb') as f: client_cert = f.read()
        with open(GRPC_CLIENT_KEY_PATH, 'rb') as f: client_key = f.read()
        with open(GRPC_CA_CERT_PATH, 'rb') as f: root_certs = f.read()

        client_call_credentials = grpc.ssl_client_credentials(
            root_certificates=root_certs,
            private_key_certificate_chain_pairs=[(client_cert, client_key)],
        )

        channel = grpc.aio.secure_channel(AUTH_SVC_ADDR, grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(root_certificates=root_certs),
            client_call_credentials
        ))
        yield auth_pb2_grpc.AuthServiceStub(channel)
    except FileNotFoundError as e:
        logger.error(f"TLS certificate file not found for gRPC client: {e}. Ensure PKI setup is complete.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="gRPC TLS certificates not found")
    except Exception as e:
        logger.error(f"Failed to create gRPC channel to Auth service: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service connection failed")
    finally:
        if channel:
            await channel.close()

# --- Database Session Dependency ---
from src.database.session import get_async_db
from src.database.crud import (
    get_user_by_id as crud_get_user_by_id,
    get_portfolio_by_id as crud_get_portfolio_by_id,
    get_user_by_email # Import if needed for signup/login API
)
from src.database.models import User, Portfolio # Import models

# --- Authentication Dependency ---
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

        db_user = await crud_get_user_by_id(db, user_id=user_id)
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
    """Extracts user ID from the authenticated User object."""
    return current_user.id

# --- API Routes ---

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "6.4.0", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/v1/auth/verify")
async def verify_token_endpoint(
    token: str, 
    auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client)
):
    """Verifies a token using the Auth gRPC service (public endpoint)."""
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token is required")
    try:
        response = await auth_client.ValidateToken(auth_pb2.TokenRequest(token=token))
        return {
            "valid": response.valid,
            "user_id": response.user_id,
            "email": response.email,
            "tier": response.tier,
            "expires_at": response.expires_at,
            "issued_at": response.issued_at,
            "token_type": response.token_type,
            "roles": response.roles
        }
    except grpc.RpcError as e:
        logger.error(f"Auth gRPC error: {e.code()} - {e.details()}")
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.details())
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during token verification")

# --- Protected Routes Example ---

@app.get("/api/v1/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: User = Depends(get_current_user) # Use the refined dependency
):
    """Returns information about the currently authenticated user."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "tier": current_user.tier,
        "roles": current_user.roles, 
        "is_verified": current_user.is_verified,
        "mfa_enabled": current_user.mfa_enabled,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
    }

# --- Include Routers ---
from api.routes.portfolio import router as portfolio_router
app.include_router(portfolio_router)

from api.routes.trade import router as trade_router
app.include_router(trade_router)

from api.routes.ml import router as ml_router
app.include_router(ml_router)

from api.routes.market_data import router as market_data_router
app.include_router(market_data_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
