import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from src.api.main import app
from src.pricing.black_scholes import BSParameters, OptionGreeks
from src.security.auth import get_current_user_flexible
from src.security.rate_limit import rate_limit
from src.utils.cache import pricing_cache, get_redis_client
import json

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_deps():
    mock_redis_client = AsyncMock()
    app.dependency_overrides[get_current_user_flexible] = lambda: {"id": "user-1"}
    app.dependency_overrides[rate_limit] = lambda: None
    app.dependency_overrides[get_redis_client] = lambda: mock_redis_client
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_redis():
    return app.dependency_overrides[get_redis_client]()

@pytest.fixture
def mock_cache():
    with patch("src.api.routes.pricing.pricing_cache") as mock_pc:
        mock_pc.get_option_price = AsyncMock(return_value=None)
        mock_pc.set_option_price = AsyncMock()
        mock_pc.get_greeks = AsyncMock(return_value=None)
        mock_pc.set_greeks = AsyncMock()
        yield mock_pc

def test_calculate_price_black_scholes(mock_cache):
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "volatility": 0.2,
        "rate": 0.05,
        "option_type": "call",
        "model": "black_scholes"
    }
    
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy") as mock_factory:
        mock_strategy = MagicMock()
        mock_strategy.price.return_value = 10.45
        mock_factory.return_value = mock_strategy
        
        response = client.post("/api/v1/pricing/price", json=payload)
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["price"] == 10.45
        assert data["cached"] is False

def test_calculate_price_cached(mock_cache):
    mock_cache.get_option_price.return_value = 10.45
    
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call",
        "model": "black_scholes"
    }
    
    response = client.post("/api/v1/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["cached"] is True
    assert response.json()["data"]["price"] == 10.45

def test_calculate_price_heston_cached(mock_redis, mock_cache):
    mock_redis.get.return_value = json.dumps({
        "timestamp": 10000000000, 
        "params": {
            "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7
        }
    })
    
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call",
        "model": "heston", "symbol": "AAPL"
    }
    
    with patch("src.api.routes.pricing.HestonModelFFT") as MockHeston:
        mock_model = MagicMock()
        mock_model.price_call.return_value = 12.0
        MockHeston.return_value = mock_model
        
        with patch("src.api.routes.pricing.time.time", return_value=10000000000):
            response = client.post("/api/v1/pricing/price", json=payload)
            assert response.status_code == 200
            assert response.json()["data"]["price"] == 12.0

def test_calculate_price_heston_missing_symbol():
    payload = {"spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "model": "heston"}
    response = client.post("/api/v1/pricing/price", json=payload)
    assert response.status_code == 422
    assert "Symbol is required" in response.json()["message"]

def test_calculate_batch_prices(mock_cache):
    payload = {
        "options": [
            {"spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"},
            {"spot": 110, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"}
        ]
    }
    
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy") as mock_factory:
        mock_strategy = MagicMock()
        mock_strategy.price.side_effect = [10.0, 15.0]
        mock_factory.return_value = mock_strategy
        
        response = client.post("/api/v1/pricing/batch", json=payload)
        assert response.status_code == 200
        results = response.json()["data"]["results"]
        assert len(results) == 2
        assert results[0]["price"] == 10.0

def test_calculate_greeks(mock_cache):
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy") as mock_factory:
        mock_strategy = MagicMock()
        mock_greeks = OptionGreeks(delta=0.6, gamma=0.02, vega=0.4, theta=-0.05, rho=0.5)
        mock_strategy.calculate_greeks.return_value = mock_greeks
        mock_strategy.price.return_value = 10.45
        mock_factory.return_value = mock_strategy
        
        response = client.post("/api/v1/pricing/greeks", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["delta"] == 0.6
        assert data["option_price"] == 10.45

def test_calculate_iv():
    payload = {
        "option_price": 10.45,
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "rate": 0.05, "option_type": "call"
    }
    
    with patch("src.api.routes.pricing.implied_volatility", return_value=0.2):
        response = client.post("/api/v1/pricing/implied-volatility", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["implied_volatility"] == 0.2

def test_calculate_exotic_price_barrier():
    payload = {
        "exotic_type": "barrier",
        "barrier_type": "up-and-out",
        "barrier": 120.0,
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    
    with patch("src.api.routes.pricing.price_exotic_option", return_value=(8.5, [8.4, 8.6])):
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 8.5

def test_calculate_exotic_price_invalid_barrier_type():
    payload = {
        "exotic_type": "barrier",
        "barrier_type": "invalid",
        "spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05,
        "option_type": "call"
    }
    response = client.post("/api/v1/pricing/exotic", json=payload)
    assert response.status_code in [400, 422]

def test_calculate_price_error_handling(mock_cache):
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call",
        "model": "black_scholes"
    }
    
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=Exception("Circuit Breaker Open")):
        response = client.post("/api/v1/pricing/price", json=payload)
        assert response.status_code == 503
        
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=ValueError("Invalid params")):
        response = client.post("/api/v1/pricing/price", json=payload)
        assert response.status_code == 422

def test_calculate_exotic_price_asian(mock_cache):
    payload = {
        "exotic_type": "asian",
        "asian_type": "arithmetic",
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call",
        "n_observations": 252
    }
    with patch("src.api.routes.pricing.price_exotic_option", return_value=(7.5, [7.4, 7.6])):
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 7.5

def test_calculate_exotic_price_lookback(mock_cache):
    payload = {
        "exotic_type": "lookback",
        "strike_type": "floating",
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    with patch("src.api.routes.pricing.price_exotic_option", return_value=(15.0, [14.5, 15.5])):
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 15.0

def test_calculate_exotic_price_digital(mock_cache):
    payload = {
        "exotic_type": "digital",
        "payout": 1.0,
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    with patch("src.api.routes.pricing.price_exotic_option", return_value=(0.6, [0.59, 0.61])):
        response = client.post("/api/v1/pricing/exotic", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 0.6

def test_calculate_price_heston_stale(mock_redis, mock_cache):
    # Mock stale data (timestamp is old)
    mock_redis.get.return_value = json.dumps({
        "timestamp": 1000, 
        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7}
    })
    
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call",
        "model": "heston", "symbol": "AAPL"
    }
    
    with patch("src.api.routes.pricing.time.time", return_value=10000), \
         patch("src.pricing.factory.PricingEngineFactory.get_strategy") as mock_factory:
        
        mock_strategy = MagicMock()
        mock_strategy.price.return_value = 10.45
        mock_factory.return_value = mock_strategy
        
        response = client.post("/api/v1/pricing/price", json=payload)
        # Should fallback to black_scholes
        assert response.status_code == 200
        assert response.headers.get("X-Pricing-Model") == "Black-Scholes-Fallback"

def test_calculate_greeks_cached(mock_cache):
    mock_greeks = OptionGreeks(delta=0.6, gamma=0.02, vega=0.4, theta=-0.05, rho=0.5)
    mock_cache.get_greeks.return_value = mock_greeks
    
    payload = {
        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    
    with patch("src.pricing.factory.PricingEngineFactory.get_strategy") as mock_factory:
        mock_strategy = MagicMock()
        mock_strategy.price.return_value = 10.45
        mock_factory.return_value = mock_strategy
        
        response = client.post("/api/v1/pricing/greeks", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["delta"] == 0.6