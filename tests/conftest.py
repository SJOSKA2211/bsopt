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

#  OPTIMIZED: Inject mocks
try:
    import tests.mock_all  # noqa: F401
except ImportError:
    try:
        import mock_all  # noqa: F401
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def startup_session():
    """Session-wide initialization with robust retries."""
    import time

    from sqlalchemy import create_engine, text

    # Ensure settings uses the TEST database URL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        is_docker = os.getenv("INSIDE_DOCKER") == "1"
        db_host = os.getenv("POSTGRES_HOST") or ("postgres" if is_docker else "localhost")
        # Use the known hex password as default if DATABASE_URL is missing
        db_url = f"postgresql://admin:29a47839acf362c9ebb5679a@{db_host}:5432/bsopt_test"

    # Force test DB name if not already there
    if "bsopt_test" not in db_url:
        if "/" in db_url:
            db_url = db_url.rsplit("/", 1)[0] + "/bsopt_test"
        else:
            db_url = db_url + "/bsopt_test"

    engine = create_engine(db_url)
    max_retries = 30
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                break
        except Exception as e:
            if i == max_retries - 1:
                pytest.exit(f"Could not connect to Postgres after {max_retries} seconds: {e}")
            time.sleep(1)

    # 3. Init Redis mock/client
    import asyncio

    from src.utils.cache import init_redis_cache

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Use a task if loop is already running
        loop.create_task(init_redis_cache())
    else:
        loop.run_until_complete(init_redis_cache())

    yield


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Ensure environment variables are set for all tests, prioritizing existing env."""
    is_docker = os.getenv("INSIDE_DOCKER") == "1"
    db_host = os.getenv("POSTGRES_HOST") or ("postgres" if is_docker else "localhost")
    redis_host = os.getenv("REDIS_HOST") or ("redis" if is_docker else "localhost")

    db_url = (
        os.getenv("DATABASE_URL")
        or f"postgresql://admin:29a47839acf362c9ebb5679a@{db_host}:5432/bsopt_test"
    )
    redis_url = os.getenv("REDIS_URL") or f"redis://{redis_host}:6379/0"

    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("JWT_SECRET", os.getenv("JWT_SECRET", "test_secret_key_change_me_in_prod"))
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")


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
