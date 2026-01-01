import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.pricing.black_scholes import BSParameters

client = TestClient(app)

@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.price.return_value = 10.5
    strategy.calculate_greeks.return_value = MagicMock(
        delta=0.5, gamma=0.05, theta=-0.01, vega=0.1, rho=0.02
    )
    return strategy

@pytest.mark.asyncio
async def test_calculate_price_success(mock_strategy):
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "spot": 100.0,
            "strike": 105.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes"
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/price", json=payload)
            assert response.status_code == 200
            response_json = response.json()
            data = response_json["data"]
            assert data["price"] == 10.5
            assert data["spot"] == 100.0

def test_calculate_price_invalid_params(mock_strategy):
    mock_strategy.price.side_effect = ValueError("Invalid spot price")
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "spot": 100.0,
            "strike": 105.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes"
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/price", json=payload)
            assert response.status_code == 422
            data = response.json()
            assert data["error"] == "ValidationError"
            assert "Invalid spot price" in data["message"]

def test_calculate_price_unexpected_error(mock_strategy):
    mock_strategy.price.side_effect = Exception("System crash")
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "spot": 100.0,
            "strike": 105.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes"
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/price", json=payload)
            assert response.status_code == 500
            data = response.json()
            assert data["error"] == "InternalServerError"
            assert "Internal error" in data["message"]

def test_calculate_price_validation_error():
    payload = {
        "strike": 105.0,
        "time_to_expiry": 0.5,
        "rate": 0.05,
        "volatility": 0.2
    }
    response = client.post("/api/v1/pricing/price", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"

def test_calculate_batch_prices_success(mock_strategy):
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "options": [
                {
                    "spot": 100.0,
                    "strike": 105.0,
                    "time_to_expiry": 0.5,
                    "rate": 0.05,
                    "volatility": 0.2,
                    "option_type": "call",
                    "model": "black_scholes"
                },
                {
                    "spot": 110.0,
                    "strike": 100.0,
                    "time_to_expiry": 1.0,
                    "rate": 0.05,
                    "volatility": 0.25,
                    "option_type": "put",
                    "model": "black_scholes"
                }
            ]
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/batch", json=payload)
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["total_count"] == 2
            assert len(data["results"]) == 2

def test_calculate_batch_prices_with_error(mock_strategy):
    mock_strategy.price.side_effect = [10.5, Exception("Price error")]
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "options": [
                {"spot": 100.0, "strike": 105.0, "time_to_expiry": 0.5, "rate": 0.05, "volatility": 0.2, "option_type": "call", "model": "black_scholes"},
                {"spot": 110.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "volatility": 0.25, "option_type": "put", "model": "black_scholes"}
            ]
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/batch", json=payload)
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["total_count"] == 1

def test_calculate_price_circuit_breaker(mock_strategy):
    mock_strategy.price.side_effect = Exception("Circuit Breaker is open")
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "spot": 100.0, "strike": 105.0, "time_to_expiry": 0.5, "rate": 0.05, "volatility": 0.2, "option_type": "call", "model": "black_scholes"
        }
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/price", json=payload)
            assert response.status_code == 503
            assert "Circuit Breaker" in response.json()["message"]

def test_calculate_greeks_success(mock_strategy):
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call"
        }
        with patch("src.api.routes.pricing.pricing_cache.get_greeks", new_callable=AsyncMock) as mock_get_cache:
            mock_get_cache.return_value = None
            response = client.post("/api/v1/pricing/greeks", json=payload)
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["delta"] == 0.5
            assert data["gamma"] == 0.05

def test_calculate_iv_success():
    with patch("src.api.routes.pricing.implied_volatility", return_value=0.25):
        payload = {
            "option_price": 10.0,
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.05,
            "option_type": "call"
        }
        response = client.post("/api/v1/pricing/implied-volatility", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["implied_volatility"] == 0.25

def test_calculate_exotic_price_success():
    with patch("src.api.routes.pricing.price_exotic_option", return_value=(15.0, [14.5, 15.5])):
        payload = {
            "exotic_type": "asian",
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "n_observations": 50,
            "asian_type": "geometric"
        }
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["price"] == 15.0
        assert data["exotic_type"] == "asian"

def test_calculate_exotic_price_barrier_missing_type():
    payload = {
        "exotic_type": "barrier", "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "volatility": 0.2, "option_type": "call"
    }
    response = client.post("/api/v1/pricing/exotic", json=payload)
    assert response.status_code == 422
    assert "barrier_type required" in response.json()["message"]

def test_calculate_price_cache_hit():
    with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = 12.34
        payload = {"spot": 100.0, "strike": 105.0, "time_to_expiry": 0.5, "rate": 0.05, "volatility": 0.2, "option_type": "call", "model": "black_scholes"}
        response = client.post("/api/v1/pricing/price", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["cached"] is True
        assert response.json()["data"]["price"] == 12.34

def test_calculate_batch_prices_cache_hit(mock_strategy):
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", return_value=mock_strategy):
        payload = {"options": [{"spot": 100.0, "strike": 105.0, "time_to_expiry": 0.5, "rate": 0.05, "volatility": 0.2, "option_type": "call", "model": "black_scholes"}]}
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = 10.5
            response = client.post("/api/v1/pricing/batch", json=payload)
            assert response.status_code == 200
            assert response.json()["data"]["cached_count"] == 1

def test_calculate_greeks_cache_hit():
    with patch("src.api.routes.pricing.pricing_cache.get_greeks", new_callable=AsyncMock) as mock_get, \
         patch("src.api.routes.pricing.PricingEngineFactory.get_strategy") as mock_factory:
        mock_get.return_value = MagicMock(delta=0.5, gamma=0.05, theta=-0.01, vega=0.1, rho=0.02)
        mock_factory.return_value.price.return_value = 10.5
        payload = {"spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "volatility": 0.2, "option_type": "call"}
        response = client.post("/api/v1/pricing/greeks", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["delta"] == 0.5

def test_calculate_greeks_error():
    with patch("src.api.routes.pricing.PricingEngineFactory.get_strategy", side_effect=Exception("Factory error")):
        payload = {"spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "volatility": 0.2, "option_type": "call"}
        response = client.post("/api/v1/pricing/greeks", json=payload)
        assert response.status_code == 422

def test_calculate_iv_value_error():
    with patch("src.api.routes.pricing.implied_volatility", side_effect=ValueError("IV error")):
        payload = {"option_price": 10.0, "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "option_type": "call"}
        response = client.post("/api/v1/pricing/implied-volatility", json=payload)
        assert response.status_code == 422

def test_calculate_iv_generic_exception():
    with patch("src.api.routes.pricing.implied_volatility", side_effect=Exception("Crash")):
        payload = {"option_price": 10.0, "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "option_type": "call"}
        response = client.post("/api/v1/pricing/implied-volatility", json=payload)
        assert response.status_code == 500

def test_calculate_exotic_price_generic_exception():
    with patch("src.api.routes.pricing.price_exotic_option", side_effect=Exception("Pricing fail")):
        payload = {"exotic_type": "asian", "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05, "volatility": 0.2, "option_type": "call"}
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 422