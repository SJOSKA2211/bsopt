import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

# Patch JSONB to use JSON for SQLite
from sqlalchemy.orm import sessionmaker

from src.database.crud import (
    create_portfolio,
    create_user,
    get_user_by_email,
    get_user_portfolios,
)
from src.database.models import Base


@pytest.fixture(autouse=True)
def patch_jsonb(monkeypatch):
    # This is a bit hacky, but common for SQLite testing of PG models
    pass


@pytest.fixture
def db_session():
    # Force SQLite to accept JSONB as JSON
    engine = create_engine("sqlite:///:memory:")

    # Define a custom compilation for JSONB on SQLite
    from sqlalchemy.ext.compiler import compiles

    @compiles(postgresql.JSONB, "sqlite")
    def compile_jsonb_sqlite(type_, compiler, **kw):
        return "JSON"

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_user_crud(db_session):
    # Create
    user = create_user(db_session, "test@example.com", "Password123!", "Test User")
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"

    # Get
    fetched = get_user_by_email(db_session, "test@example.com")
    assert fetched.id == user.id


def test_portfolio_crud(db_session):
    user = create_user(db_session, "test@example.com", "Password123!", "Test User")
    portfolio = create_portfolio(db_session, user.id, "My Portfolio", 10000.0)

    assert portfolio.name == "My Portfolio"
    assert portfolio.user_id == user.id

    portfolios = get_user_portfolios(db_session, user.id)
    assert len(portfolios) == 1
    assert portfolios[0].name == "My Portfolio"
