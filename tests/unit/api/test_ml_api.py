import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock
import httpx
import os

client = TestClient(app)

@pytest.fixture(autouse=True)
def set_testing_env():
    from src.utils.cache import get_redis_client
    
    # Create an async mock for the pipeline execution
    async def async_execute():
        return 1, []  # count, results
        
    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.execute.side_effect = async_execute
    mock_redis.pipeline.return_value = mock_pipeline
    
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    with patch.dict(os.environ, {"TESTING": "true"}):
        yield
    app.dependency_overrides.clear()

VALID_REQUEST = {
    "underlying_price": 100.0,
    "strike": 100.0,
    "time_to_expiry": 1.0,
    "is_call": 1,
    "moneyness": 1.0,
    "log_moneyness": 0.0,
    "sqrt_time_to_expiry": 1.0,
    "days_to_expiry": 365.0,
    "implied_volatility": 0.2
}

def test_proxy_predict_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "price": 10.5,
            "model_type": "xgb",
            "latency_ms": 15.0,
            "timestamp": "2026-01-17T00:00:00Z"
        }
    }
    
    with patch('httpx.AsyncClient.post', return_value=mock_response):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 10.5

def test_proxy_predict_validation_error():
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"message": "Invalid input from service"}
    
    with patch('httpx.AsyncClient.post', return_value=mock_response):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 422
        assert "Invalid input" in response.json()["message"]

def test_proxy_predict_service_unavailable():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.json.return_value = {"message": "Down for maintenance"}
    
    with patch('httpx.AsyncClient.post', return_value=mock_response):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 503
        assert "Down for maintenance" in response.json()["message"]

def test_proxy_predict_internal_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"message": "Boom"}
    
    with patch('httpx.AsyncClient.post', return_value=mock_response):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 500
        assert "Boom" in response.json()["message"]

def test_proxy_predict_request_error():
    with patch('httpx.AsyncClient.post', side_effect=httpx.RequestError("Connection failed")):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 503
        assert "unreachable" in response.json()["message"]

def test_proxy_predict_unexpected_exception():
    with patch('httpx.AsyncClient.post', side_effect=ValueError("Unexpected")):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 500
        assert "Internal error" in response.json()["message"]

def test_proxy_predict_unparseable_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.side_effect = Exception("Not JSON")
    
    with patch('httpx.AsyncClient.post', return_value=mock_response):
        response = client.post("/api/v1/ml/predict", json=VALID_REQUEST)
        assert response.status_code == 500
        assert "ML service error" in response.json()["message"]
