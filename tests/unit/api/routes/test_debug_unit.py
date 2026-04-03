from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_get_tracemalloc_snapshot_not_active():
    # Deeper bypass: patch the verify_token dependency since it's used in main.py
    # and the middleware likely relies on the request state being set.
    with (
        patch("tracemalloc.is_tracing", return_value=False),
        patch("src.api.index.verify_token", return_value={"id": "admin"}),
        patch("src.api.middleware.security.JWTAuthenticationMiddleware.dispatch") as mock_dispatch,
    ):

        async def side_effect(request, call_next):
            # Manually set user in state to satisfy get_current_user
            request.state.user = {"id": "admin"}
            return await call_next(request)

        mock_dispatch.side_effect = side_effect

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
        patch("src.api.middleware.security.JWTAuthenticationMiddleware.dispatch") as mock_dispatch,
    ):

        async def side_effect(request, call_next):
            request.state.user = {"id": "admin"}
            return await call_next(request)

        mock_dispatch.side_effect = side_effect

        response = client.get("/api/v1/debug/tracemalloc_snapshot")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "top_10_memory_allocations" in data
