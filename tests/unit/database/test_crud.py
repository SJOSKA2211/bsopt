import pytest
import pytest_asyncio
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from src.database.crud import (
    create_portfolio,
    create_user,
    get_user_by_email,
    get_user_portfolios,
)
from src.database.models import Base


# Patch JSONB for SQLite
@compiles(postgresql.JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(postgresql.INET, "sqlite")
def compile_inet_sqlite(type_, compiler, **kw):
    return "TEXT"


@pytest_asyncio.fixture
async def db_session():
    # Use aiosqlite for async SQLite
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_user_crud(db_session):
    # Create
    user = await create_user(db_session, "test@example.com", "Password123!", "Test User")
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"

    # Get
    fetched = await get_user_by_email(db_session, "test@example.com")
    assert fetched.id == user.id


@pytest.mark.asyncio
async def test_portfolio_crud(db_session):
    user = await create_user(db_session, "test@example.com", "Password123!", "Test User")
    portfolio = await create_portfolio(db_session, user.id, "My Portfolio", 10000.0)

    assert portfolio.name == "My Portfolio"
    assert portfolio.user_id == user.id

    portfolios = await get_user_portfolios(db_session, user.id)
    assert len(portfolios) == 1
    assert portfolios[0].name == "My Portfolio"
