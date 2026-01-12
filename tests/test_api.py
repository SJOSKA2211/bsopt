from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "BS-Opt API is running"}

def test_metrics_endpoint():
    # Initial request to increment counter
    client.get("/")
    
    # Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert 'endpoint="/"' in response.text
    assert 'http_status="200"' in response.text
    assert "api_request_latency_seconds" in response.text

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}