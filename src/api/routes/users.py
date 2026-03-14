"""
User Management Routes (Optimized & Async)
"""

import math

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import (
    DataResponseStruct,
    PaginatedResponseStruct,
    PaginationMetaStruct,
    SuccessResponse,
)
from src.api.schemas.user import UserResponse, UserUpdateRequest
from src.database import get_async_db, set_user_context
from src.database.models import User
from src.security.auth import get_current_user, require_tier

router = APIRouter(prefix="/users", tags=["Users"], default_response_class=MsgspecJSONResponse)


@router.get("/me")
async def get_current_user_profile(user: User = Depends(get_current_user)) -> DataResponseStruct:
    """
    Fetch the authenticated user's profile.
    """
    return DataResponseStruct(data=UserResponse.from_orm(user))


@router.patch("/me")
async def update_current_user_profile(
    update_data: UserUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Update profile for the current user (RLS Hardened).
    """
    # 1. Set RLS context
    await set_user_context(db, str(user.id))

    if update_data.full_name is not None:
        user.full_name = update_data.full_name

    try:
        db.add(user)  # In async, we ensure object is in session
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update profile") from e

    return SuccessResponse(message="Profile updated in High-Performance")


@router.get(
    "",
    response_model=None,
    dependencies=[Depends(require_tier(["admin", "enterprise"]))],
)
async def list_users(db: AsyncSession = Depends(get_async_db), page: int = 1, page_size: int = 20) -> PaginatedResponseStruct:
    """
    List users (Restricted to High-Tier/Admin).
    """
    # Bound page_size to prevent unbounded queries
    page_size = max(1, min(page_size, 100))
    page = max(1, page)

    # 1. Count total users
    count_stmt = select(func.count(User.id))
    count_result = await db.execute(count_stmt)
    total = count_result.scalar() or 0

    # 2. Fetch paginated users
    users_stmt = select(User).offset((page - 1) * page_size).limit(page_size)
    users_result = await db.execute(users_stmt)
    users = users_result.scalars().all()

    return PaginatedResponseStruct(
        items=[UserResponse.from_orm(u) for u in users],
        pagination=PaginationMetaStruct(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size),
            has_next=page * page_size < total,
            has_prev=page > 1,
        ),
    )
