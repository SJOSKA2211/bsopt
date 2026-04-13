import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.index import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "running" in response.json()["message"]


def test_health_endpoints():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 200


def test_diagnostics_imports():
    with TestClient(app) as client:
        response = client.get("/api/diagnostics/imports")
        assert response.status_code == 200
        assert "successful_imports" in response.json()


def test_http_exception_handler():
    with TestClient(app) as client:
        # Trigger HTTPException via /admin-only (missing token)
        response = client.get("/admin-only")
        assert response.status_code == 401
        assert response.json()["error"] == "http_error"


def test_admin_only_success():
    from src.auth.auth import get_current_active_user
    from src.database.models import User

    # Bypass auth by mocking the current active user with admin tier
    mock_user = User(tier="admin", is_active=True)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    with TestClient(app) as client:
        response = client.get("/admin-only")
        assert response.status_code == 200
        assert "Admin" in response.json()["message"]

    app.dependency_overrides.pop(get_current_active_user, None)


@pytest.mark.asyncio
async def test_graphql_context():
    from api.index import get_context

    request = MagicMock()

    os.environ["TESTING"] = "true"
    assert get_context(request) == {}

    os.environ["TESTING"] = "false"
    assert get_context(request) == {"request": request}

    os.environ["TESTING"] = "true"  # Reset


def test_response_time_header():
    with TestClient(app) as client:
        response = client.get("/")
        assert "X-Response-Time" in response.headers
        assert float(response.headers["X-Response-Time"]) >= 0


def test_gzip_compression():
    # Create a large response by mocking an endpoint or using one that returns lots of data
    @app.get("/large-data")
    async def large_data():
        return {"data": "x" * 2000}

    with TestClient(app) as client:
        # Standard request should be compressed if 'Accept-Encoding' is set
        response = client.get("/large-data", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == 200
        assert response.headers.get("Content-Encoding") == "gzip"