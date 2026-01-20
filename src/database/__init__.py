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
from typing import Generator, cast

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
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
_SLOW_QUERY_THRESHOLD_MS: int | None = None

def _initialize_db_components():
    global _engine, _SessionLocal, _SLOW_QUERY_THRESHOLD_MS
    if _engine is None or _SessionLocal is None:
        # Optimized for 1000+ concurrent users with PostgreSQL
        # Pool Sizing: 40 base + 110 overflow = 150 max connections
        _engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=40,
            max_overflow=110,
            pool_recycle=1800,
            pool_pre_ping=True,
            pool_use_lifo=True,
            pool_timeout=30,
            echo=settings.DEBUG if hasattr(settings, "DEBUG") else False,
            connect_args={
                "connect_timeout": 10,
                "application_name": "bsopt_api",
                "options": "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000",
            },
        )

        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
            expire_on_commit=False,
        )
        _SLOW_QUERY_THRESHOLD_MS = settings.SLOW_QUERY_THRESHOLD_MS

        # Register event listeners only once
        if not event.contains(_engine, "before_cursor_execute", before_cursor_execute):
            event.listen(_engine, "before_cursor_execute", before_cursor_execute)
        if not event.contains(_engine, "after_cursor_execute", after_cursor_execute):
            event.listen(_engine, "after_cursor_execute", after_cursor_execute)
        if not event.contains(_engine, "handle_error", handle_error):
            event.listen(_engine, "handle_error", handle_error)


# =============================================================================
# QUERY TIMING & MONITORING
# =============================================================================

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

    if _SLOW_QUERY_THRESHOLD_MS is not None and elapsed_ms > _SLOW_QUERY_THRESHOLD_MS:
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
    _initialize_db_components()
    if _SessionLocal is None:
        raise RuntimeError("SessionLocal is not initialized.")
    db = _SessionLocal()
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
    _initialize_db_components()
    if _SessionLocal is None:
        raise RuntimeError("SessionLocal is not initialized.")
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