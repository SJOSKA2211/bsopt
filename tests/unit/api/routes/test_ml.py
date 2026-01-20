import pytest
import httpx
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes.ml import router
from src.security.rate_limit import rate_limit

app = FastAPI()
app.include_router(router)

# Override rate limit for testing
app.dependency_overrides[rate_limit] = lambda: None

client = TestClient(app)

@pytest.mark.asyncio
async def test_proxy_predict_mocker(mocker):
    # Mock httpx.AsyncClient.post
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {"price": 10.5, "model_type": "xgb", "latency_ms": 5.0}, 
        "message": "success"
    }
    
    # We need to mock the context manager __aenter__
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
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
    
    response = client.post("/ml/predict", json=valid_payload)
    
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 10.5

@pytest.mark.asyncio
async def test_proxy_predict_error_400(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"message": "Invalid model"}
    
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
        "underlying_price": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "is_call": 1, "moneyness": 1.0, "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0, "days_to_expiry": 365.0, "implied_volatility": 0.2
    }
    
    response = client.post("/ml/predict", json=valid_payload)
    assert response.status_code == 422 # ValidationException maps to 422

@pytest.mark.asyncio
async def test_proxy_predict_error_503(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 503
    mock_response.json.return_value = {"message": "Overloaded"}
    
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
        "underlying_price": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "is_call": 1, "moneyness": 1.0, "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0, "days_to_expiry": 365.0, "implied_volatility": 0.2
    }
    response = client.post("/ml/predict", json=valid_payload)
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_proxy_predict_error_500(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.json.side_effect = Exception("Parse fail")
    
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
        "underlying_price": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "is_call": 1, "moneyness": 1.0, "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0, "days_to_expiry": 365.0, "implied_volatility": 0.2
    }
    response = client.post("/ml/predict", json=valid_payload)
    assert response.status_code == 500

@pytest.mark.asyncio
async def test_proxy_predict_request_error(mocker):
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(side_effect=httpx.RequestError("Conn err"))
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
        "underlying_price": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "is_call": 1, "moneyness": 1.0, "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0, "days_to_expiry": 365.0, "implied_volatility": 0.2
    }
    response = client.post("/ml/predict", json=valid_payload)
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_proxy_predict_unexpected_error(mocker):
    # Mock AsyncClient.post to raise an unexpected exception
    # We need to mock the instance returned by __aenter__
    mock_client = mocker.Mock()
    mock_client.post = mocker.AsyncMock(side_effect=RuntimeError("Unexpected"))
    mocker.patch("httpx.AsyncClient.__aenter__", return_value=mock_client)
    
    valid_payload = {
        "underlying_price": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "is_call": 1, "moneyness": 1.0, "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0, "days_to_expiry": 365.0, "implied_volatility": 0.2
    }
    response = client.post("/ml/predict", json=valid_payload)
    assert response.status_code == 500
    assert "Internal error during ML prediction" in response.json()["detail"]["message"]
