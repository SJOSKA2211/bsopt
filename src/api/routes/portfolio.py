"""
Portfolio routes backing the dashboard overview widgets.
Enhanced with God-Mode Database integration and RLS enforcement.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import DataResponse
from src.database import get_async_db, set_user_context
from src.database.models import Portfolio, Position, User
from src.security.auth import get_current_active_user

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("")
@router.get("/")
async def get_portfolio(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Return the user's primary portfolio overview including positions (RLS Hardened)."""
    # 1. Set RLS Context
    await set_user_context(db, str(current_user.id))
    
    # 2. Fetch primary portfolio (first one found)
    stmt = select(Portfolio).where(Portfolio.user_id == current_user.id).limit(1)
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        return {
            "balance": 0.0,
            "totalValue": 0.0,
            "positionsCount": 0,
            "positions": [],
            "message": "No portfolio found for user"
        }

    # 3. Fetch positions for this portfolio (Filtered by RLS)
    pos_stmt = select(Position).where(Position.portfolio_id == portfolio.id)
    pos_result = await db.execute(pos_stmt)
    positions = pos_result.scalars().all()

    return {
        "id": str(portfolio.id),
        "name": portfolio.name,
        "balance": float(portfolio.cash_balance),
        "totalValue": float(portfolio.cash_balance), # Simplified for base case
        "positionsCount": len(positions),
        "positions": [
            {
                "id": str(p.id),
                "symbol": p.symbol,
                "quantity": p.quantity,
                "entry_price": float(p.entry_price),
                "status": p.status
            }
            for p in positions
        ],
    }


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
) -> dict:
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
        status="open"
    )
    
    db.add(new_pos)
    await db.commit()
    await db.refresh(new_pos)
    
    return {"id": str(new_pos.id), "status": "position_created_solenya_tight"}


@router.delete("/positions/{position_id}")
async def delete_position(
    position_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Delete (close) a position by ID (RLS Enforced)."""
    await set_user_context(db, str(current_user.id))
    
    stmt = select(Position).where(Position.id == position_id)
    result = await db.execute(stmt)
    position = result.scalar_one_or_none()
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found or unauthorized")
        
    await db.delete(position)
    await db.commit()
    
    return {"message": "Position purged from manifold", "id": str(position_id)}
