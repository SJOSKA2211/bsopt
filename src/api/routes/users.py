"""
User Management Routes
======================

Endpoints for user profile management:
- Get/update profile
- Usage statistics
- Account management
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

import numpy as np
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from src.api.exceptions import (
    ConflictException,
    InternalServerException,
    NotFoundException,
)
from src.api.schemas.common import DataResponse, ErrorResponse, PaginatedResponse, PaginationMeta, SuccessResponse
from src.api.schemas.user import (
    APIKeyCreateRequest,
    APIKeyResponse,
    UserResponse,
    UserStatsResponse,
    UserUpdateRequest,
)
from src.config import settings
from src.database import get_db
from src.database.models import APIKey, User
from src.security.audit import AuditEvent, log_audit
from src.security.auth import (
    get_current_active_user,
    require_admin,
)
from src.utils.cache import publish_to_redis, redis_channel_updates
from src.utils.sanitization import sanitize_string
import secrets
import hashlib

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


# --- Mocking/Simulation for User Stats Update ---

def _get_mock_user_stats(user: User) -> UserStatsResponse:
    """Generates mock user stats for demonstration."""
    now = datetime.now(timezone.utc)
    return UserStatsResponse(
        total_requests=np.random.randint(1000, 5000),
        requests_today=np.random.randint(50, 200),
        requests_this_month=np.random.randint(500, 2000),
        rate_limit_remaining=settings.rate_limit_tiers.get(user.tier, 100),
        rate_limit_reset=now + timedelta(minutes=np.random.randint(5, 60)),
    )


# =============================================================================
# API Key Management
# =============================================================================

@router.get("/me/keys", response_model=DataResponse[List[APIKeyResponse]])
async def list_api_keys(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all active API keys for the current user."""
    keys = db.query(APIKey).filter(APIKey.user_id == user.id, APIKey.is_active == True).all()
    return DataResponse(data=[
        APIKeyResponse(
            id=str(k.id),
            name=k.name,
            prefix=k.prefix,
            created_at=k.created_at,
            last_used_at=k.last_used_at
        ) for k in keys
    ])

@router.post("/me/keys", response_model=DataResponse[APIKeyResponse])
async def create_api_key(
    request: APIKeyCreateRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate a new secure API key."""
    raw_key = f"bs_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:8]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    new_key = APIKey(
        user_id=user.id,
        name=request.name,
        prefix=prefix,
        key_hash=key_hash,
        is_active=True
    )
    
    db.add(new_key)
    db.commit()
    db.refresh(new_key)
    
    return DataResponse(
        data=APIKeyResponse(
            id=str(new_key.id),
            name=new_key.name,
            prefix=new_key.prefix,
            created_at=new_key.created_at,
            raw_key=raw_key
        ),
        message="API Key created. Store this securely as it will not be shown again."
    )

@router.delete("/me/keys/{key_id}", response_model=SuccessResponse)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == user.id).first()
    if not key:
        raise NotFoundException(message="API Key not found")
    
    key.is_active = False
    db.commit()
    
    return SuccessResponse(message="API Key revoked successfully")


# =============================================================================
# User Management Routes
# =============================================================================


@router.get(
    "/me",
    response_model=DataResponse[UserResponse],
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_current_user_profile(
    user: User = Depends(get_current_active_user),
):
    """ Retrieve the profile information of the currently authenticated user. """
    return DataResponse(
        data=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            tier=user.tier,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_mfa_enabled=user.is_mfa_enabled,
            created_at=user.created_at,
            last_login=user.last_login,
        )
    )


@router.patch(
    "/me",
    response_model=DataResponse[UserResponse],
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def update_current_user_profile(
    request: Request,
    update_data: UserUpdateRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """ Update the profile details of the currently authenticated user. """
    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise NotFoundException(message="User account not found")

    updated_fields = []
    if update_data.full_name is not None:
        db_user.full_name = sanitize_string(update_data.full_name)
        updated_fields.append("full_name")

    if update_data.email is not None and update_data.email != db_user.email:
        existing = db.query(User).filter(User.email == update_data.email).first()
        if existing:
            raise ConflictException(message="Email already in use")
        db_user.email = update_data.email
        db_user.is_verified = False
        updated_fields.append("email")

    if updated_fields:
        db.commit()
        db.refresh(db_user)
        log_audit(AuditEvent.USER_UPDATE, user=db_user, request=request, details={"fields": updated_fields})

    return DataResponse(data=UserResponse.from_orm(db_user))


@router.get(
    "/me/stats",
    response_model=DataResponse[UserStatsResponse],
    responses={401: {"model": ErrorResponse}},
)
async def get_user_stats(
    user: User = Depends(get_current_active_user),
):
    """ Get detailed usage statistics. """
    return DataResponse(data=_get_mock_user_stats(user))


@router.delete(
    "/me",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def delete_current_user_account(
    request: Request,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """ Deactivate account. """
    db_user = db.query(User).filter(User.id == user.id).first()
    try:
        db_user.is_active = False
        db.commit()
        log_audit(AuditEvent.USER_DELETE, user=db_user, request=request)
    except Exception as e:
        db.rollback()
        raise InternalServerException(message="Deactivation failed")

    return SuccessResponse(message="Account deactivated")


@router.get(
    "/{user_id}",
    response_model=DataResponse[UserResponse],
    dependencies=[Depends(require_admin())],
)
async def get_user_by_id(user_id: str, db: Session = Depends(get_db)):
    """ Admin: Get user by ID. """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise NotFoundException(message="User not found")
    return DataResponse(data=UserResponse.from_orm(user))


@router.get(
    "",
    response_model=PaginatedResponse[UserResponse],
    dependencies=[Depends(require_admin())],
)
async def list_users(
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
):
    """ Admin: List all users. """
    query = db.query(User)
    if search:
        query = query.filter(User.email.ilike(f"%{search}%"))
    
    total = query.count()
    users = query.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedResponse(
        items=[UserResponse.from_orm(u) for u in users],
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size),
            has_next=page * page_size < total,
            has_prev=page > 1
        )
    )