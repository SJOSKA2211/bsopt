"""
PostgreSQL Connection Management (High-Performance)
Optimized for PG16 + TimescaleDB 2.17+ with robust pooling and retry logic.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar, cast

import msgspec
import structlog
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from src.shared.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# --- SERIALIZATION ---
_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder()


def msgspec_dumps(obj: Any) -> str:
    return _encoder.encode(obj).decode()


def msgspec_loads(s: str | bytes) -> Any:
    return _decoder.decode(s)


class DatabaseManager:
    """
    High-Performance Database Manager:
    Handles sync and async engines, pooling strategies, and RLS context.
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._async_engine: AsyncEngine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._initialized = False

    def get_urls(self) -> tuple[str, str]:
        """Constructs sync and async database URLs based on environment."""
        db_url = settings.DATABASE_URL
        app_name = f"{settings.PROJECT_NAME}_{settings.ENVIRONMENT}"

        if settings.is_production and "sslmode" not in db_url:
            separator = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{separator}sslmode=require"

        #  Favor psycopg (v3) for sync path, fallback to psycopg2
        separator = "&" if "?" in db_url else "?"
        driver = "psycopg"
        try:
            import psycopg  # noqa: F401
        except ImportError:
            driver = "psycopg2"

        sync_url = f"{db_url}{separator}application_name={app_name}".replace(
            "postgresql://", f"postgresql+{driver}://"
        )

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

        return sync_url, async_url

    def _setup_events(self, engine: Engine | AsyncEngine) -> None:
        """Attaches institutional-grade performance monitoring events to the engine."""
        from src.shared.tracing import get_tracer

        tracer = get_tracer(__name__)

        def _normalize_statement(statement: str) -> str:
            """Simple normalization to group similar queries."""
            import re
            # Replace numeric literals and UUIDs with placeholders
            s = re.sub(r"'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b'", "?", statement)
            s = re.sub(r"\b\d+\b", "?", s)
            return " ".join(s.split())

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(
            conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
        ) -> None:
            conn.info.setdefault("query_start_time", []).append(time.time())
            
            # Start a manual span if tracing is enabled and we're in a trace context
            if hasattr(context, "attributes"):
                 context.attributes["db.statement.normalized"] = _normalize_statement(statement)

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(
            conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
        ) -> None:
            if not conn.info.get("query_start_time"):
                return
            
            start_time = conn.info["query_start_time"].pop()
            duration_ms = (time.time() - start_time) * 1000
            
            if duration_ms > settings.SLOW_QUERY_THRESHOLD_MS:
                normalized = _normalize_statement(statement)
                logger.warning(
                    "slow_query_detected",
                    duration_ms=round(duration_ms, 2),
                    statement_norm=normalized[:500],
                    original_sample=statement[:100] + "..." if len(statement) > 100 else statement
                )

    def initialize(self) -> None:
        """Initializes both sync and async engines with optimized settings."""
        if self._initialized:
            return

        sync_url, async_url = self.get_urls()
        app_name = f"{settings.PROJECT_NAME}_{settings.ENVIRONMENT}"

        # 1. Sync Engine Initialization
        sync_pool_kwargs: dict[str, Any]
        if settings.PGBOUNCER_ENABLED:
            logger.info("pgbouncer_detected: enabling NullPool for sync engine")
            sync_pool_class: type[NullPool] | type[QueuePool] = NullPool
            sync_pool_kwargs = {}
        else:
            sync_pool_class = QueuePool
            sync_pool_kwargs = {
                "pool_size": settings.DATABASE_MIN_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
                "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
                "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
                "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            }

        self._engine = create_engine(
            sync_url,
            poolclass=sync_pool_class,
            json_serializer=msgspec_dumps,
            json_deserializer=msgspec_loads,
            **sync_pool_kwargs,
        )
        self._setup_events(self._engine)

        # 2. Async Engine Initialization (Weaponized)
        async_pool_kwargs: dict[str, Any]
        if settings.PGBOUNCER_ENABLED:
            async_pool_class: type[NullPool] | None = NullPool
            async_pool_kwargs = {}
        else:
            # OPTIMIZED: SQLAlchemy 2.0 automatically adapts QueuePool for AsyncEngine
            # if we don't specify it explicitly.
            async_pool_class = None
            async_pool_kwargs = {
                "pool_size": settings.DATABASE_MIN_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_POOL_SIZE - settings.DATABASE_MIN_POOL_SIZE,
                "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
                "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
                "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            }

        self._async_engine = create_async_engine(
            async_url,
            poolclass=async_pool_class,
            json_serializer=msgspec_dumps,
            json_deserializer=msgspec_loads,
            connect_args={
                "ssl": (True if settings.is_production and "postgresql" in async_url else False),
                "server_settings": {
                    "application_name": app_name,
                    "tcp_keepalives_idle": "60",
                    "tcp_keepalives_interval": "10",
                    "tcp_keepalives_count": "5",
                    "statement_timeout": "600000",  # 10 minutes
                },
                "command_timeout": settings.DATABASE_POOL_TIMEOUT,
            },
            **async_pool_kwargs,
        )
        # Event listeners for async engine are slightly different in SQLAlchemy,
        # but for simplicity we use the same sync-style events which are supported for AsyncEngine's sync_engine.
        self._setup_events(self._async_engine.sync_engine)

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine, class_=AsyncSession, expire_on_commit=False
        )

        # 3. OpenTelemetry Instrumentation
        from src.shared.tracing import instrument_database

        instrument_database(self._engine)

        self._initialized = True
        logger.info("database_manager_initialized", pgbouncer=settings.PGBOUNCER_ENABLED)

    @property
    def engine(self) -> Engine:
        if not self._engine:
            self.initialize()
        return cast(Engine, self._engine)

    @property
    def async_engine(self) -> AsyncEngine:
        if not self._async_engine:
            self.initialize()
        return cast(AsyncEngine, self._async_engine)

    @property
    def session_factory(self) -> sessionmaker[Session]:
        if not self._session_factory:
            self.initialize()
        return cast(sessionmaker[Session], self._session_factory)

    @property
    def async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if not self._async_session_factory:
            self.initialize()
        return cast(async_sessionmaker[AsyncSession], self._async_session_factory)

    async def dispose(self) -> None:
        """Gracefully shuts down engines."""
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            await self._async_engine.dispose()
        self._initialized = False
        logger.info("database_engines_disposed")


db_manager = DatabaseManager()


# --- COMPATIBILITY EXPORTS ---
def get_engine() -> Engine:
    return db_manager.engine


def get_async_engine() -> AsyncEngine:
    return db_manager.async_engine


def get_sessionmaker() -> sessionmaker[Session]:
    return db_manager.session_factory


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return db_manager.async_session_factory


# Legacy Lazy Loaders
class LazySessionFactory:
    def __init__(self, getter: Any) -> None:
        self._getter = getter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._getter()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._getter(), name)


SessionLocal = LazySessionFactory(get_sessionmaker)
AsyncSessionLocal = LazySessionFactory(get_async_sessionmaker)


# --- DEPENDENCIES ---
def get_db() -> Generator[Session, None, None]:
    """Dependency for synchronous DB sessions."""
    db = db_manager.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Generator[Session, None, None]:
    return get_db()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for asynchronous DB sessions."""
    async with db_manager.async_session_factory() as session:
        yield session


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    db = db_manager.session_factory()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context() -> AsyncGenerator[AsyncSession, None]:
    async with db_manager.async_session_factory() as session:
        yield session


# --- UTILITIES ---
async def set_user_context(session: AsyncSession, user_id: str) -> None:
    """Sets the app.current_user_id in the Postgres session for RLS."""
    await session.execute(
        text("SET LOCAL app.current_user_id = :user_id"), {"user_id": str(user_id)}
    )


async def health_check() -> dict[str, Any]:
    """Enhanced database connectivity health check with retry (Asynchronous)."""
    status: dict[str, Any] = {"status": "unhealthy", "pgbouncer": settings.PGBOUNCER_ENABLED}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use async engine for health check
            async_engine = db_manager.async_engine
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                version = (await conn.execute(text("SHOW server_version"))).scalar()
                status["status"] = "healthy"
                status["version"] = version
                return status
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("database_health_check_failed", error=str(e))
                status["error"] = str(e)
            else:
                await asyncio.sleep(1)  # Async backoff
    return status


def create_tables() -> None:
    """Creates all metadata tables with optimization hooks."""
    if not settings.is_production or settings.ENVIRONMENT == "test":
        from src.database.models import Base

        engine = db_manager.engine

        # 1. Extensions & Schema (Ensuring standard compliance)
        with engine.connect() as conn:
            try:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                logger.warning("failed_to_initialize_core_extensions", error=str(e))

        # 2. Metadata Creation
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("database_metadata_synchronized")
        except Exception as e:
            logger.error("database_metadata_sync_failed", error=str(e))


async def dispose_engine() -> None:
    await db_manager.dispose()
