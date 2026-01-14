"""
This module handles Alembic environment configuration, including database
connection setup and migration script loading.
"""

import asyncio
import logging
import os
from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Import models here to ensure they are registered with Alembic's Base
from src.database.models import Base
from src.database.models import User, OptionPrice, Portfolio, Position, Order, MLModel, ModelPrediction, RateLimit, RequestLog

# Configure logging
fileConfig(context.config.config_file_name)
logger = logging.getLogger(__name__)

# Should point to your models module location
target_metadata = Base.metadata

# Store DB URL from environment or config
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set.")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Configure SQLAlchemy for async engine
async def run_migrations_async():
    """Run migrations in async mode."""
    connectable = create_async_engine(
        DATABASE_URL,
        poolclass=pool.NullPool, # Use NullPool for migrations
        echo=False # Avoid excessive logging
    )

    async with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True, # Compare column types for alterations
            render_as_batch=True # Use batch operations for efficiency
        )

        async with context.begin_transaction():
            context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode using sync engine."""
    connectable_sync = engine_from_config(
        context.config.get_section(context.config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool
    )
    context.configure(connection=connectable_sync.connect(), target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

# Alembic configuration
config = context.config

if context.is_offline_mode():
    logger.info("Running migrations in offline mode.")
    run_migrations_online()
else:
    logger.info("Running migrations in online mode (async).")
    # Ensure event loop is running for async operations
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(run_migrations_async())

if __name__ == "__main__":
    run_migrations_online()