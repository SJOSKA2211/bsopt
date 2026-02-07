import pytest
import asyncio
from uuid import uuid4
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, User, Portfolio, Position
from src.database import crud

# In-memory SQLite for high-speed testing
DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.mark.asyncio
async def test_user_lifecycle():
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as db_session:
        # Create
        email = f"test_{uuid4().hex[:8]}@example.com"
        user = await crud.create_user(db_session, email, "password123", "Test User")
        assert user.email == email
        
        # Get by ID
        fetched = await crud.get_user_by_id(db_session, user.id)
        assert fetched.id == user.id
        
        # Get by Email
        fetched_email = await crud.get_user_by_email(db_session, email)
        assert fetched_email.id == user.id
        
        # Update login
        await crud.update_user_last_login(db_session, user.id)
        await db_session.refresh(user)
        assert user.last_login is not None

    await engine.dispose()

@pytest.mark.asyncio
async def test_portfolio_operations():
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as db_session:
        user = await crud.create_user(db_session, "port@test.com", "pass", "Port User")
        
        portfolio = Portfolio(user_id=user.id, name="Main", cash_balance=10000.0)
        db_session.add(portfolio)
        await db_session.commit()
        
        # Check eager loading
        user_with_p = await crud.get_user_with_portfolios(db_session, user.id)
        assert len(user_with_p.portfolios) == 1
        assert user_with_p.portfolios[0].name == "Main"

    await engine.dispose()

@pytest.mark.asyncio
async def test_active_users_by_tier():
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as db_session:
        # Setup users with different tiers
        await crud.create_user(db_session, "u1@t.com", "p", "U1")
        
        result = await db_session.execute(
            select(User).where(User.email == "u1@t.com")
        )
        u1 = result.scalar_one()
        u1.tier = "pro"
        u1.is_active = True
        await db_session.commit()
        
        active_pro = await crud.get_active_users_by_tier(db_session, "pro")
        assert len(active_pro) >= 1
        assert active_pro[0].email == "u1@t.com"

    await engine.dispose()
