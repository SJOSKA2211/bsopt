"""
Portfolio routes backing the dashboard overview widgets.
Enhanced with God-Mode Database integration and RLS enforcement.
"""

import msgspec
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponse, SuccessResponse
from src.database import get_async_db, set_user_context
from src.database.models import Portfolio, Position, User
from src.security.auth import get_current_active_user

router = APIRouter(
    prefix="/portfolio", tags=["Portfolio"], default_response_class=MsgspecJSONResponse
)


class PositionStruct(msgspec.Struct):
    id: str
    symbol: str
    quantity: int
    entry_price: float
    status: str


class PortfolioOverview(msgspec.Struct):
    id: str
    name: str
    balance: float
    total_value: float
    positions_count: int
    positions: list[PositionStruct]
    message: str | None = None


@router.get("")
@router.get("/")
async def get_portfolio(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> PortfolioOverview:
    """Return the user's primary portfolio overview including positions (RLS Hardened)."""
    # 1. Set RLS Context
    await set_user_context(db, str(current_user.id))

    # 2. Fetch primary portfolio with eager-loaded positions (God-Mode: 1 Round Trip)
    stmt = (
        select(Portfolio)
        .options(selectinload(Portfolio.positions))
        .where(Portfolio.user_id == current_user.id)
        .limit(1)
    )
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        return PortfolioOverview(
            id="",
            name="",
            balance=0.0,
            total_value=0.0,
            positions_count=0,
            positions=[],
            message="No portfolio found for user",
        )

    positions = portfolio.positions

    return PortfolioOverview(
        id=str(portfolio.id),
        name=portfolio.name,
        balance=float(portfolio.cash_balance),
        total_value=float(portfolio.cash_balance),  # Simplified for base case
        positions_count=len(positions),
        positions=[
            PositionStruct(
                id=str(p.id),
                symbol=p.symbol,
                quantity=p.quantity,
                entry_price=float(p.entry_price),
                status=p.status,
            )
            for p in positions
        ],
    )


@router.get("/summary")
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> DataResponse:
    """Return high-level portfolio metrics via optimized view."""
    await set_user_context(db, str(current_user.id))

    from sqlalchemy import text

    # Optimized: Use our pre-aggregated materialized view (Refreshed via background task)
    stmt = text("SELECT * FROM portfolio_summary_mv WHERE user_id = :uid")
    result = await db.execute(stmt, {"uid": current_user.id})
    row = result.fetchone()

    if not row:
        return DataResponse(data={"total_positions": 0, "cash_balance": 0.0})

    return DataResponse(data=dict(row._mapping))


@router.post("/positions", status_code=201)
async def add_position(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> DataResponse:
    """Add a new position to the first available portfolio."""
    await set_user_context(db, str(current_user.id))

    # Get primary portfolio
    stmt = select(Portfolio).where(Portfolio.user_id == current_user.id).limit(1)
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(status_code=404, detail="Primary portfolio not found")

    new_pos = Position(
        portfolio_id=portfolio.id,
        symbol=payload["symbol"].upper().strip(),
        quantity=int(payload["quantity"]),
        entry_price=float(payload["entry_price"]),
        status="open",
    )

    db.add(new_pos)
    await db.commit()
    await db.refresh(new_pos)

    return DataResponse(
        data={"id": str(new_pos.id)},
        message="position_created_solenya_tight"
    )


@router.delete("/positions/{position_id}")
async def delete_position(
    position_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> SuccessResponse:
    """Delete (close) a position by ID (RLS Enforced)."""
    await set_user_context(db, str(current_user.id))

    stmt = select(Position).where(Position.id == position_id)
    result = await db.execute(stmt)
    position = result.scalar_one_or_none()

    if not position:
        raise HTTPException(status_code=404, detail="Position not found or unauthorized")

    await db.delete(position)
    await db.commit()

    return SuccessResponse(message="Position purged from manifold")
