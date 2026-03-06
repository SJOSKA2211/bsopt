import os
import sys
from pathlib import Path

import pytest

# Force the project root into sys.path
test_dir = Path(__file__).parent.absolute()
root = test_dir.parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Force src into sys.path
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Add tests dir to path for mock_all import
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

#  CLEAR LAZY IMPORT CACHE
if "src.utils.lazy_import" in sys.modules:
    import src.utils.lazy_import

    src.utils.lazy_import._failed_imports.clear()


#  OPTIMIZED: Inject mocks
try:
    import tests.mock_all  # noqa: F401
except ImportError:
    try:
        import mock_all  # noqa: F401
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Ensure environment variables are set for all tests, prioritizing existing env."""

    # Use service names 'postgres' and 'redis' if running inside docker, otherwise 'localhost'
    is_docker = os.getenv("INSIDE_DOCKER") == "1"
    db_host = "postgres" if is_docker else "localhost"
    redis_host = "redis" if is_docker else "localhost"

    # Prioritize existing env vars (injected by Docker Compose)
    db_url = os.getenv("DATABASE_URL") or f"postgresql://admin:password@{db_host}:5432/bsopt"
    redis_url = os.getenv("REDIS_URL") or f"redis://{redis_host}:6379/0"

    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("JWT_SECRET", os.getenv("JWT_SECRET", "test_secret_key_change_me_in_prod"))
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("TESTING", "true")

@pytest.fixture(scope="session", autouse=True)
def startup_session():
    """Session-wide initialization."""
    import os

    from src.config import settings

    # Ensure settings uses the TEST database URL
    test_db_url = os.getenv(
        "DATABASE_URL_TEST", "postgresql://admin:password@postgres:5432/bsopt_test"
    )
    settings.DATABASE_URL = test_db_url

    from src.database import create_tables
    from src.utils.cache import init_redis_cache

    # Wait for DB to be ready with retries
    import time

    from sqlalchemy import create_engine, text

    print(f"DEBUG: DATABASE_URL value: {settings.DATABASE_URL}")

    engine = create_engine(settings.DATABASE_URL)
    max_retries = 30
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                break
        except Exception:
            if i == max_retries - 1:
                raise
            time.sleep(1)

    # Create tables for tests
    create_tables()

    # Init Redis mock/client (running sync wrapper if needed)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(init_redis_cache())

    yield


@pytest.fixture
def api_client():
    """Returns a FastAPI TestClient."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_db_session(mocker):
    """Returns a mocked SQLAlchemy Session."""
    from sqlalchemy.orm import Session

    session = mocker.MagicMock(spec=Session)
    return session
