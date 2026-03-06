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
async def startup_session():
    """Session-wide initialization."""
    from src.database import create_tables
    from src.utils.cache import init_redis_cache

    # Create tables for tests
    create_tables()

    # Init Redis mock/client
    await init_redis_cache()

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
