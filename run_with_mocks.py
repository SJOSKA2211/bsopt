import os
import sys
import logging
from unittest.mock import MagicMock, AsyncMock

# --- 1. Environment Configuration ---
os.environ["DATABASE_URL"] = "sqlite:///./mock.db"  # Use sync sqlite for simplicity in initialization, or handle async carefully
# Actually, the app uses async engine for main app, but we can use sync for init.
# Wait, if I set sqlite:///... the app's database/__init__.py might try to create async engine with it.
# src/database/__init__.py:
# async_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")
# So "sqlite:///..." becomes "sqlite+aiosqlite:///..." which is correct for async engine.
# And "sqlite:///..." works for sync engine.
# So this is fine.

os.environ["REDIS_URL"] = "redis://mock"
os.environ["JWT_SECRET"] = "mock_secret_for_testing_only"
os.environ["ENVIRONMENT"] = "dev"
os.environ["LOG_LEVEL"] = "DEBUG"

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.getcwd())

print("DEBUG: Patching PostgreSQL dialect...")
# --- 2. Monkey Patching for SQLite Compatibility ---
# Patch UUID to work with SQLite (as String)
import sqlalchemy.types as types
from sqlalchemy.dialects import postgresql

class MockUUID(types.TypeDecorator):
    impl = types.String
    cache_ok = True

    def __init__(self, as_uuid=False, **kwargs):
        self.as_uuid = as_uuid
        super().__init__(**kwargs)

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(types.String)

    def process_bind_param(self, value, dialect):
        return str(value) if value else None

    def process_result_value(self, value, dialect):
        return value

# Apply patch before importing models
postgresql.UUID = MockUUID
postgresql.JSONB = types.JSON  # Patch JSONB to generic JSON

# Patch vector type if used (pgvector)
if not hasattr(postgresql, 'VECTOR'):
    postgresql.VECTOR = types.String
print("DEBUG: PostgreSQL dialect patched.")

# --- 3. Mocking Infrastructure Dependencies ---
print("DEBUG: Mocking Redis...")
# Mock Redis
redis_mock = MagicMock()
r_client = MagicMock()
# Async methods for redis client
r_client.get = AsyncMock(return_value=None)
r_client.set = AsyncMock(return_value=True)
r_client.exists = AsyncMock(return_value=0)
r_client.delete = AsyncMock(return_value=1)
r_client.close = AsyncMock()
r_client.ping = AsyncMock(return_value=True)

redis_mock.from_url.return_value = r_client
sys.modules["redis"] = redis_mock
sys.modules["redis.asyncio"] = redis_mock

# Mock Celery/RabbitMQ if imported
sys.modules["celery"] = MagicMock()
print("DEBUG: Infrastructure mocked.")

# --- 4. Database Initialization ---
def init_db():
    print("🚀 Initializing Mock Database (SQLite)...")
    try:
        from src.database import engine, Base
        print("DEBUG: Engine imported.")
        
        # Create tables synchronously since we are using the sync engine for DDL in dev usually
        # But here we might need to use the sync engine logic from database/__init__.py
        Base.metadata.create_all(bind=engine)
        print("✅ Tables Created")
    except Exception as e:
        print(f"ERROR in init_db: {e}")
        import traceback
        traceback.print_exc()

# --- 5. Application Startup ---
if __name__ == "__main__":
    # Initialize DB before starting app
    print("DEBUG: Starting init_db...")
    init_db()  
    
    # Import app after patching
    try:
        print("DEBUG: Importing app...")
        from src.api.main import app
        import uvicorn
        
        print(" Starting API with Mocks on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
