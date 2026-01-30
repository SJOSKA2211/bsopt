"""
User Management Routes
======================

Endpoints for user profile management:
- Get/update profile
- Usage statistics
- Account management
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.database import get_async_db
from src.database.models import APIKey, User

from fastapi import APIRouter, Depends, Request
import secrets
import hashlib
import math
from typing import List, Optional

from src.api.schemas.common import DataResponse, SuccessResponse, PaginatedResponse, PaginationMeta
from src.api.schemas.user import APIKeyResponse, APIKeyCreateRequest, UserResponse, UserUpdateRequest
from src.api.exceptions import NotFoundException, ConflictException
from src.security.auth import get_current_active_user, require_tier
from src.security.audit import AuditEvent, log_audit
from src.utils.sanitization import sanitize_string
from src.utils.cache import publish_to_redis, redis_channel_updates

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me/keys", response_model=DataResponse[List[APIKeyResponse]])
async def list_api_keys(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List active API keys (Async)."""
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == user.id, APIKey.is_active == True)
    )
    keys = result.scalars().all()
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
    db: AsyncSession = Depends(get_async_db)
):
    """Generate API key (Async)."""
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
    await db.commit()
    await db.refresh(new_key)
    
    return DataResponse(
        data=APIKeyResponse(
            id=str(new_key.id),
            name=new_key.name,
            prefix=new_key.prefix,
            created_at=new_key.created_at,
            raw_key=raw_key
        )
    )

@router.delete("/me/keys/{key_id}", response_model=SuccessResponse)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Revoke API key (Async)."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise NotFoundException(message="API Key not found")
    
    key.is_active = False
    await db.commit()
    
    return SuccessResponse(message="API Key revoked successfully")


# =============================================================================
# User Management Routes
# =============================================================================


@router.get(
    "/me",
    response_model=DataResponse[UserResponse],
)
async def get_current_user_profile(
    user: User = Depends(get_current_active_user),
):
    """Retrieve profile (Async)."""
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
)
async def update_current_user_profile(
    request: Request,
    update_data: UserUpdateRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Update profile (Async)."""
    result = await db.execute(select(User).where(User.id == user.id))
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise NotFoundException(message="User account not found")

    updated_fields = []
    if update_data.full_name is not None:
        db_user.full_name = sanitize_string(update_data.full_name)
        updated_fields.append("full_name")

    if update_data.email is not None and update_data.email != db_user.email:
        res = await db.execute(select(User).where(User.email == update_data.email))
        if res.scalar_one_or_none():
            raise ConflictException(message="Email already in use")
        db_user.email = update_data.email
        db_user.is_verified = False
        updated_fields.append("email")

    if updated_fields:
        await db.commit()
        await db.refresh(db_user)
        log_audit(AuditEvent.USER_UPDATE, user=db_user, request=request, details={"fields": updated_fields})
        await publish_to_redis(redis_channel_updates, {"event": "user_update", "user_id": str(db_user.id)})

    return DataResponse(data=UserResponse.model_validate(db_user))


@router.delete(
    "/me",
    response_model=SuccessResponse,
)
async def delete_current_user_account(
    request: Request,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Deactivate account (Async)."""
    result = await db.execute(select(User).where(User.id == user.id))
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise NotFoundException(message="User account not found")
        
    db_user.is_active = False
    await db.commit()
    log_audit(AuditEvent.USER_DELETE, user=db_user, request=request)
    await publish_to_redis(redis_channel_updates, {"event": "user_delete", "user_id": str(db_user.id)})

    return SuccessResponse(message="Account deactivated")


@router.get(
    "/{user_id}",
    response_model=DataResponse[UserResponse],
    dependencies=[Depends(require_tier(["enterprise"]))],
)
async def get_user_by_id(user_id: str, db: AsyncSession = Depends(get_async_db)):
    """Enterprise: Get user (Async)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundException(message="User not found")
    return DataResponse(data=UserResponse.model_validate(user))


@router.get(
    "",
    response_model=PaginatedResponse[UserResponse],
    dependencies=[Depends(require_tier(["enterprise"]))],
)
async def list_users(
    db: AsyncSession = Depends(get_async_db),
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    tier: Optional[str] = None,
    is_active: Optional[bool] = None,
):
    """Enterprise: List users (Async)."""
    stmt = select(User)
    if search:
        stmt = stmt.where(User.email.ilike(f"%{search}%"))
    if tier:
        stmt = stmt.where(User.tier == tier)
    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)
    
    # Count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()
    
    # Page
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    users = result.scalars().all()
    
    return PaginatedResponse(
        items=[UserResponse.model_validate(u) for u in users],
        pagination=PaginationMeta(
            total=total, page=page, page_size=page_size,
            total_pages=math.ceil(total / page_size),
            has_next=page * page_size < total,
            has_prev=page > 1
        )
    )