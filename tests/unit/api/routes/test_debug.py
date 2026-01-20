import pytest
import tracemalloc
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes.debug import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_get_tracemalloc_snapshot_inactive(mocker):
    # Mock is_tracing to False
    mocker.patch("tracemalloc.is_tracing", return_value=False)
    response = client.get("/debug/tracemalloc_snapshot")
    assert response.status_code == 500
    assert "Tracemalloc is not active" in response.json()["detail"]["message"]

def test_get_tracemalloc_snapshot_active(mocker):
    # Mock is_tracing to True
    mocker.patch("tracemalloc.is_tracing", return_value=True)
    
    # Mock take_snapshot
    mock_snapshot = mocker.Mock()
    mocker.patch("tracemalloc.take_snapshot", return_value=mock_snapshot)
    
    # Mock statistics
    mock_stat = mocker.Mock()
    mock_stat.size = 1024
    mock_stat.count = 1
    mock_frame = mocker.Mock()
    mock_frame.filename = "file.py"
    mock_frame.lineno = 10
    mock_stat.traceback = [mock_frame]
    
    mock_snapshot.statistics.return_value = [mock_stat]
    
    response = client.get("/debug/tracemalloc_snapshot")
    assert response.status_code == 200
    data = response.json()
    assert "top_10_memory_allocations" in data["data"]
    alloc = data["data"]["top_10_memory_allocations"][0]
    assert alloc["size_kb"] == 1.0
    assert alloc["traceback"][0]["file"] == "file.py"
