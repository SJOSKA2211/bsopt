"""
User Management Routes (Optimized Refactored)
"""

import math

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.schemas.common import (
    DataResponse,
    PaginatedResponse,
    PaginationMeta,
    SuccessResponse,
)
from src.api.schemas.user import UserResponse, UserUpdateRequest
from src.database import get_db
from src.database.models import User

router = APIRouter(prefix="/users", tags=["Users"])


from src.api.deps import get_current_user


@router.get("/me")
async def get_current_user_profile(user: User = Depends(get_current_user)):
    """
    Fetch the authenticated user's profile from the DB.
    """
    return DataResponse(data=UserResponse.model_validate(user))


@router.patch("/me")
async def update_current_user_profile(
    update_data: UserUpdateRequest, 
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update profile for the current user.
    """
    if update_data.full_name is not None:
        user.full_name = update_data.full_name

    db.commit()
    db.refresh(user)

    return SuccessResponse(message="Profile updated")


@router.get("", response_model=PaginatedResponse[UserResponse])
async def list_users(db: Session = Depends(get_db), page: int = 1, page_size: int = 20):
    """
    List users (Admin only logic can be added via dependency).
    """
    total = db.query(func.count(User.id)).scalar()
    users = db.query(User).offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedResponse(
        items=[UserResponse.model_validate(u) for u in users],
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size),
            has_next=page * page_size < total,
            has_prev=page > 1,
        ),
    )
