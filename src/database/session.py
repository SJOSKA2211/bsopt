from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.shared.config import settings  # Corrected import path to src.shared.config

# Database connection URL (should be loaded from environment variables)
# Example: DATABASE_URL = "postgresql+asyncpg://user:password@host:port/dbname"
DATABASE_URL = settings.DATABASE_URL

# Create an async engine
engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)

# Create a configured "Session" class
# sessionmaker returns a configured "Session" class
# For async, we use AsyncSession from sqlalchemy.ext.asyncio
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession,
)

# Dependency factory function to get a DB session
async def get_async_db():
    """Provides an asynchronous database session.
    Yields a session and ensures it's closed after use.
    """
    async with AsyncSessionLocal() as session:
        yield session

# Note: The 'settings' object is assumed to be available and configured
# to load DATABASE_URL from environment variables, as is common with pydantic-settings.
# If 'src.core.config' is not yet implemented, a placeholder or direct env var access
# might be needed for this file to be fully functional.
