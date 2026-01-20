import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.utils.circuit_breaker import pricing_circuit, db_circuit

client = TestClient(app)

def test_get_system_status():
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "circuits" in data
    assert data["circuits"]["pricing"]["state"] == "CLOSED"
