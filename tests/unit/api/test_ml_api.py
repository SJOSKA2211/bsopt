from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

@pytest.fixture
def valid_ml_payload():
    return {
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

def test_proxy_predict_success(valid_ml_payload):
    mock_response = {
        "success": True,
        "data": {
            "price": 12.5,
            "model_type": "xgb",
            "latency_ms": 10.0
        }
    }
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )
        
        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 12.5

def test_proxy_predict_service_unavailable(valid_ml_payload):
    with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("Connection failed")):
        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 503
        assert "unreachable" in response.json()["message"]

def test_proxy_predict_error_response(valid_ml_payload):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=400,
            json=lambda: {"message": "Invalid input"}
        )

        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 422
        assert "Invalid input" in response.json()["message"]

def test_proxy_predict_ml_service_503(valid_ml_payload):
    mock_response = {
        "success": False,
        "message": "ML service temporary unavailable"
    }
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=503,
            json=lambda: mock_response
        )
        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 503
        assert "ML service temporary unavailable" in response.json()["message"]

def test_proxy_predict_ml_service_generic_error(valid_ml_payload):
    mock_response = {
        "success": False,
        "message": "Something unexpected happened in ML service"
    }
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=500, # Generic error
            json=lambda: mock_response
        )
        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 500
        assert "Something unexpected happened in ML service" in response.json()["message"]

def test_proxy_predict_unexpected_error(valid_ml_payload):
    with patch("httpx.AsyncClient.post", side_effect=Exception("Unhandled error")):
        response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
        assert response.status_code == 500
        assert "Internal error during ML prediction" in response.json()["message"]