"""
User Management Routes (Singularity Refactored)
"""

import math
from typing import List, Optional
from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from src.database import get_db
from src.auth.security import RoleChecker
from src.database.models import User
from src.api.schemas.user import UserResponse, UserUpdateRequest
from src.api.schemas.common import (
    DataResponse,
    SuccessResponse,
    PaginatedResponse,
    PaginationMeta,
)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me")
async def get_current_user_profile(request: Request):
    """
    Fetch the authenticated user's profile.
    Uses request.state.user populated by the global verify_token dependency.
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")

    return DataResponse(data=request.state.user)


@router.patch("/me")
async def update_current_user_profile(
    update_data: UserUpdateRequest, request: Request, db: Session = Depends(get_db)
):
    """
    Update profile for the current user.
    """
    user_payload = request.state.user
    user_id = user_payload.get("sub")

    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if update_data.full_name is not None:
        db_user.full_name = update_data.full_name

    db.commit()
    db.refresh(db_user)

    return SuccessResponse(message="Profile updated")


@router.get(
    "",
    response_model=PaginatedResponse[UserResponse],
    dependencies=[Depends(RoleChecker(["admin"]))],
)
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
