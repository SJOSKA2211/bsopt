"""
Tests for the /api/diagnostics/imports endpoint.
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app AFTER patching
from api.index import app

client = TestClient(app)


def test_get_import_diagnostics():
    response = client.get("/api/diagnostics/imports")
    assert response.status_code == 200
    data = response.json()

    assert data["successful_imports"] == 5
    assert data["failed_imports"] == 1
    assert data["total_import_time_seconds"] == pytest.approx(0.12345)
    assert len(data["slowest_imports"]) == 2
    assert data["slowest_imports"][0]["module"] == "src.ml.HeavyModel"
    assert data["slowest_imports"][0]["duration_ms"] == pytest.approx(50.0)
    assert "src.ml.BrokenModule" in data["failures"]
    assert data["failures"]["src.ml.BrokenModule"] == "ModuleNotFoundError"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}