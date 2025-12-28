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
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.api.schemas.common import SuccessResponse
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


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: User = Depends(get_current_active_user),
):
    """
    Get current user's profile information.
    """
    return UserResponse(
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


@router.patch("/me", response_model=UserResponse)
async def update_current_user_profile(
    request: Request,
    update_data: UserUpdateRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Update current user's profile.
    """
    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    updated_fields = []
    if update_data.full_name is not None:
        db_user.full_name = update_data.full_name
        updated_fields.append("full_name")

    if update_data.email is not None and update_data.email != db_user.email:
        existing = db.query(User).filter(User.email == update_data.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already in use",
            )
        db_user.email = update_data.email
        db_user.is_verified = False  # Require re-verification
        updated_fields.append("email")

    db.commit()
    db.refresh(db_user)

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
    logger.info(f"Broadcasted user profile update for user ID: {db_user.id}")

    return UserResponse.from_orm(db_user)


@router.get("/me/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Get usage statistics and rate limit information for the current user.
    """
    # In production, this would fetch real metrics from Prometheus/Redis or DB.
    # For demonstration, we return simulated stats based on user tier.
    return _get_mock_user_stats(user)


@router.delete("/me", response_model=SuccessResponse)
async def delete_current_user_account(
    request: Request,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Delete current user's account.

    This is a soft delete - marks account as inactive.
    """
    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.is_active = False
    db.commit()

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
    logger.info(f"Broadcasted user status update for user ID: {db_user.id}")

    return SuccessResponse(
        success=True,
        message="Account has been deactivated. Contact support to permanently delete your data.",
        data={},
    )


# =============================================================================
# Admin Endpoints (Enterprise tier only)
# =============================================================================


@router.get(
    "/{user_id}", response_model=UserResponse, dependencies=[Depends(require_tier(["enterprise"]))]
)
async def get_user_by_id(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Get user by ID (admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse.from_orm(user)


# ... (rest of the main.py code including lifespan, other routers, etc.)
