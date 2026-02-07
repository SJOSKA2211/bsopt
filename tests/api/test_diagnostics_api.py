"""
Tests for the /api/diagnostics/imports endpoint.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# Mock src.utils.lazy_import and its functions
@pytest.fixture(autouse=True)
def mock_lazy_import_module():
    with patch("src.api.main.get_import_stats") as mock_get_import_stats:
        mock_get_import_stats.return_value = {
            "successful_imports": 5,
            "failed_imports": 1,
            "total_import_time": 0.12345,
            "slowest_imports": [
                ("src.ml.HeavyModel", 0.05),
                ("src.pricing.QuantumPricer", 0.03),
            ],
            "failures": {"src.ml.BrokenModule": "ModuleNotFoundError"},
        }
        yield


# Import the FastAPI app AFTER patching
from src.api.main import app

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
