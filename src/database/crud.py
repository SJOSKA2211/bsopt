from typing import Any
from uuid import UUID

from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from src.database.models import MLModel, Portfolio, Trade, User

# --- Password Hashing Context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- User CRUD ---
async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Retrieves a user by their email address."""
    stmt = select(User).filter(User.email == email)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    """Retrieves a user by their UUID."""
    stmt = select(User).filter(User.id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user_in: dict[str, Any]) -> User:
    """Creates a new user with a hashed password."""
    hashed_password = pwd_context.hash(user_in["password"])

    user_data = {
        "email": user_in["email"],
        "hashed_password": hashed_password,
        "full_name": user_in.get("full_name"),
        "tier": user_in.get("tier", "free"),
        "roles": user_in.get("roles", []),
        "is_verified": user_in.get("is_verified", False),
        "mfa_enabled": user_in.get("mfa_enabled", False),
    }
    db_user = User(**user_data)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# --- Portfolio CRUD ---
async def get_portfolio_by_id(db: AsyncSession, portfolio_id: UUID, user_id: UUID) -> Portfolio | None:
    """Retrieves a portfolio by its UUID and user UUID."""
    stmt = select(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_portfolios_for_user(db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 100) -> list[Portfolio]:
    """Retrieves all portfolios for a given user UUID."""
    stmt = select(Portfolio).filter(Portfolio.user_id == user_id).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_portfolio(db: AsyncSession, portfolio_in: dict[str, Any]) -> Portfolio:
    """Creates a new portfolio."""
    db_portfolio = Portfolio(**portfolio_in)
    db.add(db_portfolio)
    await db.commit()
    await db.refresh(db_portfolio)
    return db_portfolio

async def update_portfolio(db: AsyncSession, db_portfolio: Portfolio, portfolio_in: dict[str, Any]) -> Portfolio:
    """Updates an existing portfolio."""
    for field, value in portfolio_in.items():
        if hasattr(db_portfolio, field):
            setattr(db_portfolio, field, value)
    await db.commit()
    await db.refresh(db_portfolio)
    return db_portfolio

# --- Trade CRUD ---
async def get_trade_by_id(db: AsyncSession, trade_id: UUID) -> Trade | None:
    """Retrieves a trade by its UUID."""
    stmt = select(Trade).filter(Trade.id == trade_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_trades_for_portfolio(db: AsyncSession, portfolio_id: UUID, skip: int = 0, limit: int = 100) -> list[Trade]:
    """Retrieves all trades for a given portfolio UUID."""
    stmt = select(Trade).filter(Trade.portfolio_id == portfolio_id).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_trade(db: AsyncSession, trade_in: dict[str, Any]) -> Trade:
    """Creates a new trade."""
    db_trade = Trade(**trade_in)
    db.add(db_trade)
    await db.commit()
    await db.refresh(db_trade)
    return db_trade

# Add other CRUD functions as needed.
