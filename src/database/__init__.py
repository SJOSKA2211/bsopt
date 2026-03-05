"""
Database Session Management (Native PostgreSQL)
"""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine,
    AsyncAdaptedQueuePool
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import settings

from .models import Base

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Use optimized pooling for PostgreSQL 16
POOL_CLASS = QueuePool
ASYNC_POOL_CLASS = AsyncAdaptedQueuePool

# Ensure SSL for production environments
db_url = settings.DATABASE_URL
if settings.is_production and "sslmode" not in db_url:
    separator = "&" if "?" in db_url else "?"
    db_url = f"{db_url}{separator}sslmode=require"

# Ensure sync engine gets sync URL
sync_url = db_url.replace("+asyncpg", "")


# --- ENGINES ---
def get_engine():
    """Returns the synchronous SQLAlchemy engine."""
    return engine


def get_async_engine():
    """Returns the asynchronous SQLAlchemy engine."""
    return async_engine


# Optimized Sync Engine
engine = create_engine(
    sync_url,
    poolclass=POOL_CLASS,
    pool_size=settings.DATABASE_MIN_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,
    pool_recycle=1800,  # Recycle connections after 30 minutes
)

async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
if "sqlite" in db_url:
    async_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")

# Strip sslmode for asyncpg (it uses 'ssl' arg instead)
if "postgresql" in async_url and "?" in async_url:
    base, _ = async_url.split("?", 1)
    async_url = base

# Optimized Async Engine
async_engine = create_async_engine(
    async_url,
    poolclass=ASYNC_POOL_CLASS,
    pool_size=settings.DATABASE_MIN_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args=(
        {"ssl": True} if settings.is_production and "postgresql" in async_url else {}
    ),
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

# --- DEPENDENCIES ---


def get_session():
    """Alias for get_db for compatibility."""
    return get_db()


def get_db() -> Generator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def set_user_context(session: AsyncSession, user_id: str):
    """Sets the app.current_user_id in the Postgres session for RLS."""
    # STABLE: uses our optimized RLS function/policy
    await session.execute(
        text("SET LOCAL app.current_user_id = :user_id"), {"user_id": str(user_id)}
    )


@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context():
    async with AsyncSessionLocal() as session:
        yield session


# --- UTILITIES ---


def health_check() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"database_health_check_failed: {e}")
        return False


def create_tables():
    # Only create if not in prod/prod-like unless specifically needed
    if not settings.is_production or settings.ENVIRONMENT == "test":
        Base.metadata.create_all(bind=engine)
        logger.info("database_tables_created")


def dispose_engine():
    engine.dispose()
    logger.info("database_engine_disposed")
