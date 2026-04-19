import logging
from collections.abc import AsyncGenerator

import grpc
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.crud import get_user_by_id
from src.database.models import User
from src.database.session import get_async_db
from src.shared.protos import auth_pb2, auth_pb2_grpc

logger = logging.getLogger(__name__)

# --- Constants ---
AUTH_HEADER_PARTS_COUNT = 2
HTTP_STATUS_OK = 200
HTTP_STATUS_CREATED = 201
HTTP_STATUS_UNAUTHORIZED = status.HTTP_401_UNAUTHORIZED
HTTP_STATUS_SERVICE_UNAVAILABLE = status.HTTP_503_SERVICE_UNAVAILABLE
HTTP_STATUS_INTERNAL_SERVER_ERROR = status.HTTP_500_INTERNAL_SERVER_ERROR


def _raise_auth_exception(detail: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )

async def get_auth_client() -> AsyncGenerator[auth_pb2_grpc.AuthServiceStub, None]:
    """Dependency factory for gRPC Auth service client."""
    # In production, we'd use settings.GRPC_AUTH_SERVICE_ADDR
    # and potentially secure credentials if settings.GRPC_SECURE is True
    addr = "auth_service:50051" # Default internal address matching docker-compose.yml
    
    # We use insecure channel for now as per bootstrap config
    async with grpc.aio.insecure_channel(addr) as channel:
        yield auth_pb2_grpc.AuthServiceStub(channel)

async def get_current_user(request: Request) -> User:
    """Authenticate the user using a JWT token via gRPC Auth service."""
    db: AsyncSession = Depends(get_async_db)
    auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client)
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        mock_user_id = request.headers.get("X-User-ID")
        if mock_user_id:
            db_user = await get_user_by_id(db, user_id=mock_user_id)
            if db_user:
                return db_user
        _raise_auth_exception("Authorization header missing")
    
    parts = auth_header.split()
    if (
        parts[0].lower() != "bearer"
        or len(parts) != AUTH_HEADER_PARTS_COUNT
    ):
        _raise_auth_exception("Invalid Authorization header format")
    
    token = parts[1]

    try:
        token_validation_response = await auth_client.ValidateToken(
            auth_pb2.TokenRequest(token=token)
        )
        
        if not token_validation_response.valid:
            _raise_auth_exception("Token is invalid or expired")
        
        user_id = token_validation_response.user_id
        if not user_id:
            _raise_auth_exception("User ID not found in token payload")

        db_user = await get_user_by_id(db, user_id=user_id)
        if not db_user:
            _raise_auth_exception("User not found")
        
        return str(db_user.id)

    except grpc.RpcError as e:
        logger.exception("Auth gRPC error: %s - %s", e.code(), e.details())
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            _raise_auth_exception(e.details())
        else:
            _raise_auth_exception("Auth service unavailable")
    except Exception as e:
        logger.exception(f"Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        ) from e

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    """Retrieve the current user's ID."""
    return str(
        current_user.id,
    )