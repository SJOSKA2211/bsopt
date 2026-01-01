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
from contextlib import contextmanager
from typing import Generator, cast

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import settings

from .models import Base

logger = logging.getLogger(__name__)

# =============================================================================
# CONNECTION POOL CONFIGURATION
# =============================================================================
# Optimized for 1000+ concurrent users with PostgreSQL
#
# Pool Size Calculation:
#   - Expected concurrent requests: 1000
#   - Average query time: 10-50ms
#   - Formula: pool_size = (concurrent_requests / (1000 / avg_query_ms)) * 1.5
#   - With 50ms avg: pool_size = (1000 / 20) * 1.5 = 75
#   - We use 40 base + 110 overflow = 150 max connections
# =============================================================================

engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    # Pool sizing
    pool_size=40,  # Base connections (permanent)
    max_overflow=110,  # Additional connections under load (max 150 total)
    # Connection lifecycle
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_pre_ping=True,  # Verify connection health before use
    pool_use_lifo=True,  # Reuse most recent connections (better cache locality)
    pool_timeout=30,  # Wait up to 30s for available connection
    # Debug mode
    echo=settings.DEBUG if hasattr(settings, "DEBUG") else False,
    # PostgreSQL-specific settings
    connect_args={
        "connect_timeout": 10,
        "application_name": "bsopt_api",
        "options": "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000",
    },
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Prevent lazy loading after commit
)


# =============================================================================
# QUERY TIMING & MONITORING
# =============================================================================

# Threshold for slow query logging (milliseconds)
SLOW_QUERY_THRESHOLD_MS = settings.SLOW_QUERY_THRESHOLD_MS


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record query start time for timing."""
    conn.info.setdefault("query_start_time", []).append(time.perf_counter())


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries and record metrics."""
    start_times = conn.info.get("query_start_time", [])
    if not start_times:
        return

    elapsed_ms = (time.perf_counter() - start_times.pop()) * 1000

    if elapsed_ms > SLOW_QUERY_THRESHOLD_MS:
        # Truncate long queries for logging
        query_preview = statement[:200] + "..." if len(statement) > 200 else statement
        logger.warning(
            f"Slow query ({elapsed_ms:.1f}ms): {query_preview}",
            extra={"query_time_ms": elapsed_ms, "query": statement[:500]},
        )


@event.listens_for(Engine, "handle_error")
def handle_error(exception_context):
    """Log database errors with context."""
    logger.error(f"Database error: {exception_context.original_exception}", exc_info=True)


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for scripts and background tasks.
    Automatically commits on success, rolls back on error.

    Usage:
        with get_db_context() as db:
            db.add(item)
            # Auto-commits if no exception
    """
    db = SessionLocal()
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
    return SessionLocal()


# =============================================================================
# CONNECTION POOL UTILITIES
# =============================================================================


def get_pool_status() -> dict:
    """
    Get current connection pool statistics.

    Returns:
        dict with pool metrics for monitoring
    """
    pool = cast(QueuePool, engine.pool)
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
    try:
        with engine.connect() as conn:
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
    engine.dispose()
    logger.info("Database connection pool disposed")


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================


def create_tables():
    """
    Create all tables defined in models.
    Only use for development/testing - use Alembic for production.
    """
    if settings.ENVIRONMENT in ["dev", "test"]:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")
    else:
        logger.warning(f"Attempted to create tables in {settings.ENVIRONMENT} environment. Operation blocked.")


def drop_tables():
    """
    Drop all tables. USE WITH EXTREME CAUTION.
    """
    if settings.ENVIRONMENT in ["dev", "test"]:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped!")
    else:
        logger.warning(f"Attempted to drop tables in {settings.ENVIRONMENT} environment. Operation blocked.")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "engine",
    "SessionLocal",
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
