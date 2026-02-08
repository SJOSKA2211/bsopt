import sys
from pathlib import Path

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

# 🚀 CLEAR LAZY IMPORT CACHE
if "src.utils.lazy_import" in sys.modules:
    import src.utils.lazy_import
    src.utils.lazy_import._failed_imports.clear()

import importlib.util  # noqa: E402

# 🚀 SINGULARITY: Inject mocks
try:
    import tests.mock_all
except ImportError:
    try:
        import mock_all
    except ImportError:
        pass

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Ensure environment variables are set for all tests."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("JWT_SECRET", "test_secret_key_change_me_in_prod")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
