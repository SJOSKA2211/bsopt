import logging
from typing import AsyncGenerator

import grpc
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.crud import get_user_by_id
from src.database.models import User
from src.database.session import get_async_db
from src.shared.config import settings
from src.shared.protos import auth_pb2, auth_pb2_grpc

logger = logging.getLogger(__name__)

async def get_auth_client() -> AsyncGenerator[auth_pb2_grpc.AuthServiceStub, None]:
    """Dependency factory for gRPC Auth service client."""
    # In production, we'd use settings.GRPC_AUTH_SERVICE_ADDR
    # and potentially secure credentials if settings.GRPC_SECURE is True
    addr = "auth:50051" # Default internal address
    
    # We use insecure channel for now as per bootstrap config
    async with grpc.aio.insecure_channel(addr) as channel:
        yield auth_pb2_grpc.AuthServiceStub(channel)

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client),
) -> User:
    """
    Authenticates the user using a JWT token via gRPC Auth service.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        # For development/mock purposes if needed, check X-User-ID
        mock_user_id = request.headers.get("X-User-ID")
        if mock_user_id:
            db_user = await get_user_by_id(db, user_id=mock_user_id)
            if db_user:
                return db_user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
    parts = auth_header.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    
    token = parts[1]

    try:
        # Validate token via gRPC
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
        logger.error(f"Auth gRPC error: {e.code()} - {e.details()}")
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.details())
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    except Exception as e:
        logger.exception(f"Unexpected authentication error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during authentication")

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    """Dependency to retrieve the current user's ID."""
    return str(current_user.id)
