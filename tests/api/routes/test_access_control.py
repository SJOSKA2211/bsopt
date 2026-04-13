from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app
from src.auth.auth import get_current_active_user

client = TestClient(app)


@pytest.fixture
def mock_user_free():
    u = MagicMock()
    u.tier = "free"
    u.is_active = True
    return u


@pytest.fixture
def mock_user_enterprise():
    u = MagicMock()
    u.tier = "enterprise"
    u.is_active = True
    return u


@pytest.fixture
def mock_user_admin():
    u = MagicMock()
    u.tier = "admin"
    u.is_active = True
    return u


def test_system_health_deep_unauthorized():
    """Verify /system/health/deep requires auth and high tier."""
    # 1. No token
    response = client.get("/api/v1/system/health/deep")
    assert response.status_code == 401


def test_system_diagnostics_tier_check(mock_user_free, mock_user_enterprise):
    """Verify tier check bug fix in system.py (list vs string)."""

    # 1. Free tier should be 403
    app.dependency_overrides[get_current_active_user] = lambda: mock_user_free
    response = client.get("/api/v1/system/diagnostics/db")
    assert response.status_code == 403

    # 2. Enterprise tier should be 200 (if CRUD is mocked)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user_enterprise
    with patch("api.routes.system.crud.get_system_health_dashboard") as m_crud:
        m_crud.return_value = {"status": "ok"}
        response = client.get("/api/v1/system/diagnostics/db")
        assert response.status_code == 200

    app.dependency_overrides.clear()


def test_ml_predict_auth_required():
    """Verify ML routes now require authentication."""
    response = client.post("/api/v1/ml/predict", json={})
    assert response.status_code == 401


def test_ml_admin_routes(mock_user_free, mock_user_admin):
    """Verify ML admin routes specifically require admin/enterprise."""

    # 1. Free tier
    app.dependency_overrides[get_current_active_user] = lambda: mock_user_free
    response = client.post("/api/v1/ml/retrain")
    assert response.status_code == 403

    # 2. Admin tier
    app.dependency_overrides[get_current_active_user] = lambda: mock_user_admin
    with patch("api.routes.ml.check_threshold_and_retrain_task") as m_task:
        m_task.delay.return_value = MagicMock(id="task_123")
        response = client.post("/api/v1/ml/retrain")
        assert response.status_code == 201

    app.dependency_overrides.clear()


def test_options_chain_auth_required():
    """Verify options chain requires authentication."""
    response = client.get("/api/v1/options/chain")
    assert response.status_code == 401


def test_pricing_calculate_auth_required():
    """Verify pricing calculation requires authentication."""
    response = client.post(
        "/api/v1/pricing/calculate", json={"s": 100, "k": 100, "t": 1, "r": 0.05, "sigma": 0.2}
    )
    assert response.status_code == 401