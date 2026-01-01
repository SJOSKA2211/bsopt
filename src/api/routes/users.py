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
from datetime import datetime, timedelta, timezone

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
    UserResponse,
    UserStatsResponse,
    UserUpdateRequest,
)
from src.config import settings
from src.database import get_db
from src.database.models import User
from src.security.audit import AuditEvent, log_audit
from src.security.auth import (
    get_current_active_user,
    require_tier,
)
from src.utils.cache import publish_to_redis, redis_channel_updates
from src.utils.sanitization import sanitize_string
from typing import Optional
import math

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


# --- Mocking/Simulation for User Stats Update ---
# In a real application, these stats would be updated by actual request
# logs or other backend processes.
# For demonstration, we'll simulate updates via a dedicated endpoint.


# Helper to get user stats (mimics backend logic for /users/me/stats)
def _get_mock_user_stats(user: User) -> UserStatsResponse:
    """Generates mock user stats for demonstration."""
    now = datetime.now(timezone.utc)
    return UserStatsResponse(
        total_requests=np.random.randint(1000, 5000),
        requests_today=np.random.randint(50, 200),
        requests_this_month=np.random.randint(500, 2000),
        rate_limit_remaining=settings.rate_limit_tiers.get(user.tier, 100),  # Use actual tier logic
        rate_limit_reset=now + timedelta(minutes=np.random.randint(5, 60)),  # Random reset time
    )


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
    """
    Retrieve the profile information of the currently authenticated user.
    """
    return DataResponse(
        data=UserResponse(
            id=str(user.id),
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
    """
    Update the profile details of the currently authenticated user.

    Updating the **email** will require re-verification.
    """
    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise NotFoundException(
            message="User account not found in database",
        )

    updated_fields = []
    if update_data.full_name is not None:
        db_user.full_name = sanitize_string(update_data.full_name)
        updated_fields.append("full_name")

    if update_data.email is not None and update_data.email != db_user.email:
        existing = db.query(User).filter(User.email == update_data.email).first()
        if existing:
            raise ConflictException(
                message="The requested email is already in use by another account",
            )
        db_user.email = update_data.email
        db_user.is_verified = False  # Require re-verification
        updated_fields.append("email")

    if not updated_fields:
        return DataResponse(data=UserResponse.from_orm(db_user))

    try:
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user profile: {e}")
        raise InternalServerException(
            message="Failed to persist profile updates",
        )

    log_audit(
        AuditEvent.USER_UPDATE,
        user=db_user,
        request=request,
        details={"updated_fields": updated_fields},
    )

    # --- Broadcast user profile update ---
    user_profile_update = {
        "type": "user_profile_update",
        "payload": UserResponse.from_orm(db_user).model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(publish_to_redis(redis_channel_updates, user_profile_update))

    return DataResponse(data=UserResponse.from_orm(db_user))


@router.get(
    "/me/stats",
    response_model=DataResponse[UserStatsResponse],
    responses={401: {"model": ErrorResponse}},
)
async def get_user_stats(
    user: User = Depends(get_current_active_user),
):
    """
    Get detailed usage statistics and remaining rate limits for the current user.
    """
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
    """
    Deactivate the current user's account.

    This performs a **soft delete**, marking the account as inactive. 
    Authenticated requests will no longer be possible once completed.
    """
    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise NotFoundException(message="User not found")

    try:
        db_user.is_active = False
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error deactivating user: {e}")
        raise InternalServerException(
            message="Failed to deactivate account",
        )

    log_audit(
        AuditEvent.USER_DELETE,
        user=db_user,
        request=request,
    )

    # Broadcast user status change
    user_status_update = {
        "type": "user_status_update",
        "payload": {
            "user_id": str(db_user.id),
            "is_active": False,
            "status": "deactivated",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(publish_to_redis(redis_channel_updates, user_status_update))

    return SuccessResponse(
        success=True,
        message="Account has been deactivated successfully.",
        data={"user_id": str(db_user.id)},
    )


@router.get(
    "/{user_id}",
    response_model=DataResponse[UserResponse],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    dependencies=[Depends(require_tier(["enterprise"]))],
)
async def get_user_by_id(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve user information by user ID. 

    **Restricted to Enterprise/Admin users only.**
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise NotFoundException(
            message=f"User with ID {user_id} not found",
        )

    return DataResponse(data=UserResponse.from_orm(user))


@router.get(
    "",
    response_model=PaginatedResponse[UserResponse],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
    dependencies=[Depends(require_tier(["enterprise"]))],
)
async def list_users(
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    tier: Optional[str] = None,
    is_active: Optional[bool] = None,
):
    """
    List all users with pagination and filtering.

    **Restricted to Enterprise/Admin users only.**
    """
    query = db.query(User)

    if search:
        query = query.filter(
            (User.email.ilike(f"%{search}%")) | (User.full_name.ilike(f"%{search}%"))
        )

    if tier:
        query = query.filter(User.tier == tier)

    if is_active is not None:
        query = query.filter(User.is_active == is_active)

    total = query.count()
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    users = query.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedResponse(
        items=[UserResponse.from_orm(u) for u in users],
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


# ... (rest of the main.py code including lifespan, other routers, etc.)
