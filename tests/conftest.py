import sys
import os
from pathlib import Path

# Force the project root into sys.path
root = Path(__file__).parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Force src into sys.path
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import pytest

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Ensure environment variables are set for all tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("JWT_SECRET", "test_secret_key_change_me_in_prod")
    monkeypatch.setenv("ENVIRONMENT", "test")
