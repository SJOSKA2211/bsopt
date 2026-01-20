"""
Tests for the /api/v1/system endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import os

client = TestClient(app)

@pytest.fixture(autouse=True)
def set_testing_env():
    with patch.dict(os.environ, {"TESTING": "true"}):
        yield

from unittest.mock import patch

def test_get_system_status():
    with patch('src.api.routes.system.pricing_circuit') as mock_pricing:
        with patch('src.api.routes.system.db_circuit') as mock_db:
            # Mock states
            mock_pricing.state.value = "closed"
            mock_pricing.failure_count = 0
            mock_db.state.value = "closed"
            mock_db.failure_count = 0
            
            response = client.get("/api/v1/system/status")
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "operational"
            assert data["circuits"]["pricing"]["state"] == "closed"
            assert data["circuits"]["database"]["state"] == "closed"
