from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.database.models import User
from src.security.auth import get_current_active_user

client = TestClient(app)

# Mock user for dependency override
async def mock_get_current_active_user():
    mock_user = MagicMock(spec=User)
    mock_user.id = "admin"
    mock_user.email = "admin@example.com"
    mock_user.tier = "enterprise"
    mock_user.is_active = True
    return mock_user

@pytest.mark.asyncio
async def test_get_tracemalloc_snapshot_not_active():
    app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

    with patch("tracemalloc.is_tracing", return_value=False):
        response = client.get("/api/v1/debug/tracemalloc_snapshot")
        assert response.status_code == 500
        assert "not active" in response.json()["message"]

    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_get_tracemalloc_snapshot_success():
    app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

    mock_snapshot = MagicMock()
    mock_stat = MagicMock()
    mock_stat.size = 1024
    mock_stat.count = 1
    mock_frame = MagicMock()
    mock_frame.filename = "test.py"
    mock_frame.lineno = 1
    mock_stat.traceback = [mock_frame]
    mock_snapshot.statistics.return_value = [mock_stat]

    with (
        patch("tracemalloc.is_tracing", return_value=True),
        patch("tracemalloc.take_snapshot", return_value=mock_snapshot),
    ):
        response = client.get("/api/v1/debug/tracemalloc_snapshot")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "top_memory_allocations" in data

    app.dependency_overrides = {}
