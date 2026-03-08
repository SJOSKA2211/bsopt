import time
import msgspec
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from src.config import settings

logger = structlog.get_logger(__name__)

# --- SERIALIZATION ---
_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder()

def msgspec_dumps(obj):
    return _encoder.encode(obj).decode()

def msgspec_loads(s):
    return _decoder.decode(s)


# --- ENGINE STATE ---
_engine: Engine | None = None
_async_engine: AsyncEngine | None = None
_SessionLocal: sessionmaker | None = None
_AsyncSessionLocal: async_sessionmaker | None = None


# --- CONFIGURATION ---
def get_db_urls():
    """Constructs sync and async database URLs based on environment."""
    db_url = settings.DATABASE_URL
    app_name = f"{settings.PROJECT_NAME}_{settings.ENVIRONMENT}"

    if settings.is_production and "sslmode" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=require"

    # 🚀 GOD-MODE: Favor psycopg (v3) for sync path
    separator = "&" if "?" in db_url else "?"
    sync_url = f"{db_url}{separator}application_name={app_name}".replace("postgresql://", "postgresql+psycopg://")
    
    # Async path favors asyncpg
    async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

    if "sqlite" in db_url:
        async_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")

    # Strip sslmode for asyncpg (handled via connect_args)
    if "postgresql" in async_url and "?" in async_url:
        parts = async_url.split("?")
        base = parts[0]
        params = [p for p in parts[1].split("&") if not p.startswith("sslmode=")]
        async_url = f"{base}?{'&'.join(params)}" if params else base

    # AUTO-DETECT PGBOUNCER
    if "pgbouncer" in db_url.lower() and not settings.PGBOUNCER_ENABLED:
        logger.info("pgbouncer_auto_detected_enabling_optimization")
        settings.PGBOUNCER_ENABLED = True

    return sync_url, async_url


# --- PERFORMANCE MONITORING ---
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault("query_start_time", []).append(time.time())


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = (time.time() - conn.info["query_start_time"].pop()) * 1000
    if total_time > settings.SLOW_QUERY_THRESHOLD_MS:
        logger.warning(
            "slow_query_detected",
            duration_ms=round(total_time, 2),
            statement=statement[:500],
        )


# --- ENGINE INITIALIZATION ---
def get_engine() -> Engine:
    """Returns the synchronous SQLAlchemy engine, initializing if necessary."""
    global _engine
    if _engine is None:
        sync_url, _ = get_db_urls()

        # Dynamic Pool Configuration
        if settings.PGBOUNCER_ENABLED:
            logger.info("pgbouncer_detected: enabling NullPool for transaction mode")
            pool_class = NullPool
            pool_kwargs = {}
        else:
            pool_class = QueuePool
            pool_kwargs = {
                "pool_size": settings.DATABASE_MIN_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
                "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
                "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
                "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            }

        # ⚡ SERIALIZATION WEAPONIZATION
        _engine = create_engine(
            sync_url, 
            poolclass=pool_class, 
            json_serializer=msgspec_dumps,
            json_deserializer=msgspec_loads,
            **pool_kwargs
        )
        logger.info("sync_engine_initialized", driver="psycopg3", pgbouncer=settings.PGBOUNCER_ENABLED)

    return _engine


def get_async_engine() -> AsyncEngine:
    """Returns the asynchronous SQLAlchemy engine, initializing if necessary."""
    global _async_engine
    if _async_engine is None:
        _, async_url = get_db_urls()

        # Optimized Async Config
        pool_kwargs = {}
        app_name = f"{settings.PROJECT_NAME}_{settings.ENVIRONMENT}"

        if not settings.PGBOUNCER_ENABLED:
            pool_kwargs = {
                "pool_size": settings.DATABASE_MIN_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
                "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
                "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
                "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            }
        else:
            pool_kwargs = {"poolclass": NullPool}

        # ⚡ ASYNC WEAPONIZATION
        _async_engine = create_async_engine(
            async_url,
            json_serializer=msgspec_dumps,
            json_deserializer=msgspec_loads,
            connect_args={
                "ssl": (True if settings.is_production and "postgresql" in async_url else False),
                "server_settings": {
                    "application_name": app_name,
                    "tcp_keepalives_idle": "60",
                    "tcp_keepalives_interval": "10",
                    "tcp_keepalives_count": "5",
                },
                "statement_cache_size": 20,
                "prepared_statement_cache_size": 20,
                "command_timeout": settings.DATABASE_POOL_TIMEOUT,
            },
            **pool_kwargs,
        )

        logger.info("async_engine_initialized", pgbouncer=settings.PGBOUNCER_ENABLED)

    return _async_engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_async_sessionmaker() -> async_sessionmaker:
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _AsyncSessionLocal


# --- COMPATIBILITY EXPORTS ---
# These allow existing code to import SessionLocal/AsyncSessionLocal
# Note: They will trigger engine initialization on first use via the getters below.
class LazySessionFactory:
    def __init__(self, getter):
        self._getter = getter

    def __call__(self, *args, **kwargs):
        return self._getter()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._getter(), name)


SessionLocal = LazySessionFactory(get_sessionmaker)
AsyncSessionLocal = LazySessionFactory(get_async_sessionmaker)


# --- DEPENDENCIES ---
def get_db() -> Generator[Session, None, None]:
    """Dependency for synchronous DB sessions."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


def get_session():
    """Alias for get_db for compatibility."""
    return get_db()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for asynchronous DB sessions."""
    async with get_async_sessionmaker()() as session:
        yield session


async def set_user_context(session: AsyncSession, user_id: str):
    """Sets the app.current_user_id in the Postgres session for RLS."""
    await session.execute(
        text("SET LOCAL app.current_user_id = :user_id"), {"user_id": str(user_id)}
    )


@contextmanager
def get_db_context():
    """Context manager for synchronous DB sessions."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context():
    """Context manager for asynchronous DB sessions."""
    async with get_async_sessionmaker()() as session:
        yield session


# --- UTILITIES ---
def health_check() -> dict[str, Any]:
    """Enhanced database connectivity health check."""
    status = {"status": "unhealthy", "pgbouncer": settings.PGBOUNCER_ENABLED}
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            version = conn.execute(text("SHOW server_version")).scalar()
            status["status"] = "healthy"
            status["version"] = version
        return status
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        status["error"] = str(e)
        return status


def create_tables():
    """Creates all metadata tables if not in production."""
    if not settings.is_production or settings.ENVIRONMENT == "test":
        from sqlalchemy import text

        from src.database.models import Base

        engine = get_engine()
        with engine.connect() as conn:
            # 0. In test mode, we might want a clean slate?
            # For now, just ensure extensions and types exist.

            # 1. Enable extensions
            try:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                # pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                logger.warning("failed_to_enable_extensions", error=str(e))

            # 2. Create ENUM types manually if they don't exist
            enums = [
                ("user_tier", ["free", "pro", "enterprise"]),
                ("order_side", ["buy", "sell"]),
                (
                    "order_status",
                    ["pending", "filled", "partially_filled", "cancelled", "rejected"],
                ),
                ("order_type", ["market", "limit", "stop", "stop_limit"]),
                ("position_status", ["open", "closed"]),
                ("option_type", ["call", "put"]),
                (
                    "ml_algorithm",
                    ["xgboost", "lightgbm", "neural_network", "random_forest", "svm", "ensemble"],
                ),
            ]
            for name, values in enums:
                try:
                    # Case-insensitive check
                    res = conn.execute(
                        text("SELECT 1 FROM pg_type WHERE typname = :name"), {"name": name}
                    )
                    if not res.fetchone():
                        vals = ", ".join([f"'{v}'" for v in values])
                        conn.execute(text(f"CREATE TYPE {name} AS ENUM ({vals})"))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"failed_to_create_enum_{name}", error=str(e))

        # 3. Create Tables
        # metadata.create_all is idempotent for tables, but not for all types
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("database_tables_created")
        except Exception as e:
            logger.error("database_metadata_creation_failed", error=str(e))


async def dispose_engine():
    """Disposes of the engines, releasing resources. (Async-safe)"""
    global _engine, _async_engine
    if _engine:
        _engine.dispose()
        _engine = None

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None

    logger.info("database_engines_disposed")
