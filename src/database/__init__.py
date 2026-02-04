"""
Database Session Management (Neon Native)
"""

import logging
import time
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from src.config import settings
from .models import Base

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Use NullPool for serverless if in production to avoid connection pinning
POOL_CLASS = NullPool if settings.ENVIRONMENT == "prod" else QueuePool

# Ensure SSL for Neon
db_url = settings.DATABASE_URL
if "sslmode" not in db_url:
    separator = "&" if "?" in db_url else "?"
    db_url = f"{db_url}{separator}sslmode=require"

# --- ENGINES ---
engine = create_engine(
    db_url,
    poolclass=POOL_CLASS,
    pool_size=5 if POOL_CLASS == QueuePool else 0,
    max_overflow=10 if POOL_CLASS == QueuePool else 0,
    pool_pre_ping=True
)

async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
if "sqlite" in db_url:
    async_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")

# Strip sslmode for asyncpg (it uses 'ssl' arg instead)
if "postgresql" in async_url and "?" in async_url:
    base, _ = async_url.split("?", 1)
    async_url = base

async_engine = create_async_engine(
    async_url,
    poolclass=NullPool,
    connect_args={"ssl": True} if settings.ENVIRONMENT == "prod" and "postgresql" in async_url else {}
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# --- DEPENDENCIES ---

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

from contextlib import asynccontextmanager

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
    if settings.ENVIRONMENT in ["dev", "test"]:
        Base.metadata.create_all(bind=engine)
        logger.info("database_tables_created")

def dispose_engine():
    engine.dispose()
    logger.info("database_engine_disposed")
