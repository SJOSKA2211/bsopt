import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes.system import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_get_system_status():
    response = client.get("/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "circuits" in data
    assert "pricing" in data["circuits"]
    assert "database" in data["circuits"]
