import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import tracemalloc

client = TestClient(app)

def test_get_tracemalloc_snapshot_not_active():
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    
    response = client.get("/api/v1/debug/tracemalloc_snapshot")
    assert response.status_code == 500
    assert "not active" in response.json()["message"]

def test_get_tracemalloc_snapshot_success():
    tracemalloc.start()
    try:
        response = client.get("/api/v1/debug/tracemalloc_snapshot")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "top_10_memory_allocations" in data
        assert len(data["top_10_memory_allocations"]) <= 20
    finally:
        tracemalloc.stop()
