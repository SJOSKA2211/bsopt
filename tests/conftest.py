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
    """Session-wide initialization with robust retries."""
    import os
    import time

    import sqlalchemy
    print(f"DEBUG: sqlalchemy is mock: {hasattr(sqlalchemy, '__mock_name__') or 'mock' in str(type(sqlalchemy))}")
    print(f"DEBUG: sqlalchemy path: {sqlalchemy.__file__ if hasattr(sqlalchemy, '__file__') else 'N/A'}")

    from sqlalchemy import create_engine, text

    from src.config import settings

    # Ensure settings uses the TEST database URL
    test_db_url = os.getenv(
        "DATABASE_URL_TEST", "postgresql://admin:password@postgres:5432/bsopt_test"
    )
    # Patch settings.DATABASE_URL
    settings.DATABASE_URL = test_db_url

    print(f"DEBUG: Initializing tests with DB: {settings.DATABASE_URL}")

    # 1. Wait for Postgres to accept connections
    engine = create_engine(test_db_url, pool_pre_ping=True)
    max_retries = 120
    connected = False
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                connected = True
                print("DEBUG: Postgres is ready.")
                break
        except Exception as e:
            if i % 10 == 0:
                print(f"DEBUG: Waiting for Postgres... ({e})")
            time.sleep(1)

    if not connected:
        pytest.exit("Could not connect to Postgres after 120 seconds", returncode=1)

    # 2. Database tables are already created by init-scripts in docker-compose
    # Skip create_tables() to save time and avoid concurrent creation issues
    print("DEBUG: Skipping redundant create_tables().")

    # 3. Init Redis mock/client
    import asyncio

    from src.utils.cache import init_redis_cache
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(init_redis_cache())
    print("DEBUG: Redis cache initialized.")

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
