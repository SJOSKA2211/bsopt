from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app
from api.middleware.jwt_validator import require_auth

client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture(autouse=True)
def override_auth():
    mock_claims = MagicMock()
    mock_claims.tier = "admin"
    app.dependency_overrides[require_auth] = lambda: mock_claims
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_tracemalloc_snapshot_not_active():
    with patch("tracemalloc.is_tracing", return_value=False):
        response = client.get("/api/v1/debug/tracemalloc_snapshot")
        assert response.status_code == 500
        assert "not active" in response.json()["message"]


@pytest.mark.asyncio
async def test_get_tracemalloc_snapshot_success():
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

