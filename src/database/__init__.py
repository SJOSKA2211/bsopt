"""
Database Session Management with Optimized Connection Pooling

This module provides:
- Connection pooling optimized for 1000+ concurrent users
- Query timing and slow query logging
- Context managers for FastAPI and background tasks
- Connection health monitoring
"""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Generator, AsyncGenerator, cast, Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import settings

from .models import Base

logger = logging.getLogger(__name__)

# =============================================================================
# CONNECTION POOL CONFIGURATION (Lazy Initialization)
# =============================================================================

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_async_engine: Any | None = None
_AsyncSessionLocal: async_sessionmaker | None = None
_SLOW_QUERY_THRESHOLD_MS: int | None = None

def _initialize_db_components():
    global _engine, _SessionLocal, _async_engine, _AsyncSessionLocal, _SLOW_QUERY_THRESHOLD_MS
    
    # Sync Engine
    if _engine is None or _SessionLocal is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=50,
            max_overflow=150,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=settings.DEBUG if hasattr(settings, "DEBUG") else False,
        )
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    # Async Engine
    if _async_engine is None or _AsyncSessionLocal is None:
        # Convert DATABASE_URL to asyncpg if needed
        async_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        _async_engine = create_async_engine(
            async_url,
            pool_size=50,
            max_overflow=150,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
        )
        _AsyncSessionLocal = async_sessionmaker(
            bind=_async_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )

    _SLOW_QUERY_THRESHOLD_MS = settings.SLOW_QUERY_THRESHOLD_MS


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """Sync session dependency."""
    _initialize_db_components()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async session dependency for FastAPI."""
    _initialize_db_components()
    async with _AsyncSessionLocal() as session:
        yield session

@asynccontextmanager
async def get_async_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for workers."""
    _initialize_db_components()
    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Sync context manager."""
    _initialize_db_components()
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_session() -> Session:
    """
    Get a new session directly (caller MUST manage lifecycle, including closing the session).
    
    WARNING: This function bypasses FastAPI's dependency injection session management.
    Mismanagement (e.g., forgetting to call session.close()) can lead to connection leaks.

    Usage:
        session = get_session()
        try:
            # use session
            session.commit()
        finally:
            session.close()
    """
    _initialize_db_components()
    if _SessionLocal is None:
        raise RuntimeError("SessionLocal is not initialized.")
    return _SessionLocal()


def get_engine() -> Engine:
    """Expose the database engine, initializing components if necessary."""
    _initialize_db_components()
    if _engine is None:
        raise RuntimeError("Database engine is not initialized.")
    return _engine


# =============================================================================
# CONNECTION POOL UTILITIES
# =============================================================================


def get_pool_status() -> dict:
    """
    Get current connection pool statistics.

    Returns:
        dict with pool metrics for monitoring
    """
    _initialize_db_components()
    if _engine is None:
        raise RuntimeError("Database engine is not initialized.")
    pool = cast(QueuePool, _engine.pool)
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
        "invalidated": getattr(pool, "_invalidate_time", 0),
        "max_overflow": getattr(pool, "_max_overflow", 0),
        "total_connections": pool.size() + pool.overflow(),
    }


def health_check() -> bool:
    """
    Verify database connectivity.

    Returns:
        True if database is reachable, False otherwise
    """
    _initialize_db_components()
    if _engine is None:
        return False # Or raise an error
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def dispose_engine():
    """
    Dispose of all pooled connections.
    Call during graceful shutdown.
    """
    _initialize_db_components()
    if _engine:
        _engine.dispose()
        logger.info("Database connection pool disposed")


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================


def create_tables():
    """
    Create all tables defined in models.
    Only use for development/testing - use Alembic for production.
    """
    _initialize_db_components()
    if settings.ENVIRONMENT in ["dev", "test"]:
        if _engine is None:
            raise RuntimeError("Database engine is not initialized.")
        Base.metadata.create_all(bind=_engine)
        logger.info("Database tables created")
    else:
        logger.warning(f"Attempted to create tables in {settings.ENVIRONMENT} environment. Operation blocked.")


def drop_tables():
    """
    Drop all tables. USE WITH EXTREME CAUTION.
    """
    _initialize_db_components()
    if settings.ENVIRONMENT in ["dev", "test"]:
        if _engine is None:
            raise RuntimeError("Database engine is not initialized.")
        Base.metadata.drop_all(bind=_engine)
        logger.warning("All database tables dropped!")
    else:
        logger.warning(f"Attempted to drop tables in {settings.ENVIRONMENT} environment. Operation blocked.")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "Base",
    # Session management
    "get_db",
    "get_db_context",
    "get_session",
    # Utilities
    "get_pool_status",
    "health_check",
    "dispose_engine",
    # Models (re-export for convenience)
    "create_tables",
    "drop_tables",
]