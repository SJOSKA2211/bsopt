import os
import sys
from pathlib import Path

import pytest
import structlog
from datetime import datetime, UTC, timedelta

# Set testing environment variables before any imports
os.environ["BSOPT_ALLOW_WEAK_SECRETS"] = "true"
os.environ["ENVIRONMENT"] = "test"
os.environ["TESTING"] = "true"
os.environ["LOG_SAMPLING_RATE"] = "1.0"
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Default fallback credentials for local testing
os.environ["REDIS_PASSWORD"] = "test_redis_password_v2"
os.environ["JWT_SECRET"] = "test_secret_key_v2_must_be_long_enough_for_security"

# Force the project root into sys.path
test_dir = Path(__file__).parent.absolute()
root = test_dir.parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Force src into sys.path
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

def pytest_collection_modifyitems(items):
    """Automatically mark tests based on their directory."""
    for item in items:
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        if "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

@pytest.fixture(scope="session", autouse=True)
def startup_session():
    """Session-wide initialization."""
    structlog.get_logger().info("test_session_start")
    yield
    structlog.get_logger().info("test_session_end")

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Ensure environment variables are set for all tests, prioritizing existing env."""
    is_docker = os.getenv("INSIDE_DOCKER") == "1"
    db_host = os.getenv("POSTGRES_HOST") or ("postgres" if is_docker else "localhost")
    redis_host = os.getenv("REDIS_HOST") or ("redis" if is_docker else "localhost")

    db_url = (
        os.getenv("DATABASE_URL")
        or f"postgresql+asyncpg://admin:password@{db_host}:5432/bsopt_test"
    )
    redis_url = os.getenv("REDIS_URL") or f"redis://:{os.getenv('REDIS_PASSWORD')}@{redis_host}:6379/0"

    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("REDIS_URL", redis_url)

@pytest.fixture
def api_client(request):
    """Returns a FastAPI TestClient targeting the real app with NO mocks."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text
    from api.index import app
    from src.shared.config import settings

    # Clean DB state for integration tests
    if "integration" in request.node.nodeid:
        try:
            # Use sync engine for truncation
            sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
            engine = create_engine(sync_url)
            with engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE users CASCADE"))
                conn.commit()
        except Exception as e:
            structlog.get_logger().warning("db_truncation_failed", error=str(e))

    with TestClient(app) as client:
        yield client

@pytest.fixture
def test_user_token():
    """Generates a real, valid JWT token for a test user."""
    from src.auth.core.tokens import token_service
    
    token_pair = token_service.create_token_pair(
        user_id="test-integration-user",
        email="test@manifold.test",
        tier="admin",
        scopes=["admin", "read", "write"]
    )
    return token_pair.access_token

@pytest.fixture
def auth_headers(test_user_token):
    """Standardized Authorization headers for test requests."""
    return {"Authorization": f"Bearer {test_user_token}"}

@pytest.fixture
async def db_session():
    """Provides a real async database session."""
    from src.database import get_async_db
    async for session in get_async_db():
        yield session

@pytest.fixture
def self_healing_retry(request):
    """Optional self-healing wrapper."""
    try:
        yield
    except Exception as e:
        if "self_heal" in request.keywords:
            structlog.get_logger().error("test_failed_triggering_self_heal", error=str(e))
        raise e