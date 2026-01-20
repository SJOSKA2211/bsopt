import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from src.security.rate_limit import rate_limit

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_rate_limit():
    app.dependency_overrides[rate_limit] = lambda: None
    yield
    app.dependency_overrides = {}

VALID_ML_PAYLOAD = {
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

def test_predict_success():
    mock_response = {
        "data": {
            "price": 10.5,
            "model_type": "xgb",
            "latency_ms": 15.0
        },
        "status": "success"
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )
        
        response = client.post("/api/v1/ml/predict?model_type=xgb", json=VALID_ML_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 10.5

def test_predict_ml_service_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Test 400 -> Should raise ValidationException (422 in FastAPI)
        mock_post.return_value = MagicMock(
            status_code=400,
            json=lambda: {"message": "Invalid params"}
        )
        response = client.post("/api/v1/ml/predict", json=VALID_ML_PAYLOAD)
        assert response.status_code == 422 
        
        # Test 503
        mock_post.return_value = MagicMock(
            status_code=503,
            json=lambda: {"message": "Down"}
        )
        response = client.post("/api/v1/ml/predict", json=VALID_ML_PAYLOAD)
        assert response.status_code == 503

def test_predict_connection_error():
    with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("Connection failed")):
        response = client.post("/api/v1/ml/predict", json=VALID_ML_PAYLOAD)
        assert response.status_code == 503
        assert "unreachable" in response.json()["message"]