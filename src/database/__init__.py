from contextlib import asynccontextmanager

import structlog
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings

logger = structlog.get_logger(__name__)

# --- Async Engine (Primary) ---
async_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_MIN_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_POOL_SIZE,
    pool_pre_ping=True,  # Critical for long-lived connections
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


async def get_async_db():
    """Dependency for FastAPI (Async)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# --- Sync Engine (Legacy/Migrations/Celery) ---
sync_engine = create_engine(
    settings.DATABASE_URL.replace("+asyncpg", ""),  # Fallback to sync driver
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


def get_db():
    """Dependency for synchronous contexts."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Direct session factory for non-DI contexts."""
    return SessionLocal()


@asynccontextmanager
async def get_async_db_context():
    """Async context manager for DB access outside FastAPI dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Import base at end to avoid circular imports during init
from .models import Base as Base  # noqa: E402
