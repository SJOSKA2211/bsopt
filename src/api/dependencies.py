"""Dependencies for the API."""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

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

def _raise_auth_exception(detail: str) -> None:
    """Raise an 401 Unauthorized exception."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )

from src.shared.grpc_manager import grpc_manager

async def get_auth_client() -> auth_pb2_grpc.AuthServiceStub:
    """Provide a gRPC Auth service client using persistent pooled connections."""
    channel = await grpc_manager.get_auth_channel()
    return auth_pb2_grpc.AuthServiceStub(channel)

async def _get_token_from_header(auth_header: str | None) -> str:
    """Extract the Bearer token from the Authorization header."""
    if not auth_header:
        _raise_auth_exception("Authorization header missing")
    
    parts = auth_header.split()
    if len(parts) != AUTH_HEADER_PARTS_COUNT or parts[0].lower() != "bearer":
        _raise_auth_exception("Invalid Authorization header format")
    
    return parts[1]

async def get_current_user(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_db)],
    auth_client: Annotated[auth_pb2_grpc.AuthServiceStub, Depends(get_auth_client)],
) -> User:
    """Authenticate the user using a JWT token via gRPC Auth service."""
    auth_header = request.headers.get("Authorization")
    
    # Handle development mock bypass
    if not auth_header:
        mock_user_id = request.headers.get("X-User-ID")
        if mock_user_id:
            import uuid
            try:
                user_uuid = uuid.UUID(mock_user_id)
            except ValueError:
                _raise_auth_exception("Invalid X-User-ID format")
            db_user = await get_user_by_id(db, user_id=user_uuid)
            if db_user:
                return db_user
        _raise_auth_exception("Authorization header missing")
    
    token = await _get_token_from_header(auth_header)

    try:
        response = await auth_client.ValidateToken(
            auth_pb2.TokenRequest(token=token),
        )
        
        if not response.valid:
            _raise_auth_exception("Token is invalid or expired")
        
        if not response.user_id:
            _raise_auth_exception("User ID not found in token payload")

        import uuid
        try:
            user_uuid = uuid.UUID(response.user_id)
        except ValueError:
            _raise_auth_exception("Invalid user ID format")

        db_user = await get_user_by_id(db, user_id=user_uuid)
        if not db_user:
            _raise_auth_exception("User not found")
        
        return db_user

    except grpc.RpcError as e:
        logger.exception("Auth gRPC error: %s - %s", e.code(), e.details())
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            _raise_auth_exception(e.details())
        _raise_auth_exception("Auth service unavailable")
    except Exception:
        logger.exception("Unexpected authentication error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        ) from None

async def get_current_user_id(
    current_user: Annotated[User, Depends(get_current_user)],
) -> str:
    """Retrieve the current user's ID as a string."""
    return str(current_user.id)
