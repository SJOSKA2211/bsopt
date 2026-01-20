import pytest
import json
import time
from fastapi.testclient import TestClient
from fastapi import FastAPI, Response
from src.api.routes.pricing import router
from src.security.auth import get_current_user_flexible
from src.security.rate_limit import rate_limit
from src.utils.cache import pricing_cache, get_redis_client
from src.pricing.models import BSParameters, OptionGreeks

app = FastAPI()
app.include_router(router)

# Mock dependencies
app.dependency_overrides[get_current_user_flexible] = lambda: {"id": "test"}
app.dependency_overrides[rate_limit] = lambda: None

client = TestClient(app)

@pytest.mark.asyncio
async def test_calculate_price_standard(mocker):
    # Mock cache MISS
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=None)
    mocker.patch("src.utils.cache.pricing_cache.set_option_price", return_value=True)
    
    # Mock PricingEngine
    mock_engine = mocker.Mock()
    mock_engine.price.return_value = 10.0
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", return_value=mock_engine)
    
    app.dependency_overrides[get_redis_client] = lambda: mocker.Mock()
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 10.0

@pytest.mark.asyncio
async def test_calculate_price_cache_hit(mocker):
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=15.0)
    app.dependency_overrides[get_redis_client] = lambda: mocker.Mock()
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 15.0
    assert response.json()["data"]["cached"] is True

@pytest.mark.asyncio
async def test_calculate_price_heston_success(mocker):
    mock_redis = mocker.Mock()
    # Ensure it's an AsyncMock for await redis_client.get(cache_key)
    mock_redis.get = mocker.AsyncMock(return_value=json.dumps({
        "timestamp": time.time(),
        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.1, "rho": -0.7}
    }))
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "heston", "symbol": "AAPL"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.headers["X-Pricing-Model"] == "Heston-FFT"

@pytest.mark.asyncio
async def test_calculate_price_heston_stale_fallback(mocker):
    mock_redis = mocker.Mock()
    # 20 minutes ago
    mock_redis.get = mocker.AsyncMock(return_value=json.dumps({
        "timestamp": time.time() - 1200,
        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.1, "rho": -0.7}
    }))
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    
    # Mock BS fallback
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=10.0)
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "heston", "symbol": "AAPL"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"

def test_calculate_batch_prices(mocker):
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=None)
    mock_engine = mocker.Mock()
    mock_engine.price.return_value = 10.0
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", return_value=mock_engine)
    
    payload = {
        "options": [
            {"spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"},
            {"spot": 100, "strike": 110, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"}
        ]
    }
    response = client.post("/pricing/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["data"]["results"]) == 2

@pytest.mark.asyncio
async def test_calculate_greeks(mocker):
    mocker.patch("src.utils.cache.pricing_cache.get_greeks", return_value=None)
    mock_engine = mocker.Mock()
    mock_engine.calculate_greeks.return_value = OptionGreeks(0.5, 0.1, -0.05, 0.2, 0.1)
    mock_engine.price.return_value = 10.0
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", return_value=mock_engine)
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/greeks", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["delta"] == 0.5

def test_calculate_iv():
    payload = {
        "option_price": 10.0, "spot": 100, "strike": 100, "time_to_expiry": 1,
        "rate": 0.05, "dividend_yield": 0.0, "option_type": "call"
    }
    response = client.post("/pricing/implied-volatility", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["implied_volatility"] > 0

def test_calculate_exotic_barrier():
    payload = {
        "exotic_type": "barrier", "barrier_type": "up-and-out", "barrier": 120.0,
        "spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/exotic", json=payload)
    assert response.status_code == 200
    assert "price" in response.json()["data"]

def test_calculate_exotic_asian(mocker):
    # Mock price_exotic_option to return (price, [low, high])
    mocker.patch("src.api.routes.pricing.price_exotic_option", return_value=(5.0, [4.9, 5.1]))
    
    payload = {
        "exotic_type": "asian", "asian_type": "arithmetic", "n_observations": 12,
        "spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/exotic", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 5.0
    assert isinstance(response.json()["data"]["confidence_interval"], list)

def test_calculate_price_heston_no_symbol():
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "heston"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_calculate_price_invalid_params(mocker):
    app.dependency_overrides[get_redis_client] = lambda: mocker.Mock()
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=ValueError("Invalid params"))
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_calculate_price_circuit_breaker(mocker):
    app.dependency_overrides[get_redis_client] = lambda: mocker.Mock()
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=Exception("Circuit Breaker is OPEN"))
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 503

def test_calculate_iv_value_error(mocker):
    mocker.patch("src.api.routes.pricing.implied_volatility", side_effect=ValueError("NR fail"))
    payload = {"option_price": 10.0, "spot": 100, "strike": 100, "time_to_expiry": 1, "rate": 0.05, "option_type": "call"}
    response = client.post("/pricing/implied-volatility", json=payload)
    assert response.status_code == 422

def test_calculate_exotic_invalid_barrier():
    payload = {
        "exotic_type": "barrier", "barrier_type": "invalid", 
        "spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/exotic", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_calculate_price_heston_error_fallback(mocker):
    mock_redis = mocker.Mock()
    # Malformed JSON triggers exception
    mock_redis.get = mocker.AsyncMock(return_value=b"{invalid}")
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=10.0)
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "heston", "symbol": "AAPL"
    }
    response = client.post("/pricing/price", json=payload)
    assert response.status_code == 200
    assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"

def test_calculate_batch_prices_error(mocker):
    mocker.patch("src.utils.cache.pricing_cache.get_option_price", return_value=None)
    # Trigger exception in loop
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=Exception("Loop fail"))
    
    payload = {
        "options": [{"spot": 100, "strike": 100, "time_to_expiry": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call", "model": "black_scholes"}]
    }
    response = client.post("/pricing/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["data"]["results"]) == 0

def test_calculate_iv_unexpected_error(mocker):
    mocker.patch("src.api.routes.pricing.implied_volatility", side_effect=Exception("Crash"))
    payload = {"option_price": 10.0, "spot": 100, "strike": 100, "time_to_expiry": 1, "rate": 0.05, "option_type": "call"}
    response = client.post("/pricing/implied-volatility", json=payload)
    assert response.status_code == 500

def test_calculate_exotic_error(mocker):
    mocker.patch("src.api.routes.pricing.price_exotic_option", side_effect=Exception("Exotic fail"))
    payload = {
        "exotic_type": "asian", "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/exotic", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_calculate_greeks_error(mocker):
    mocker.patch("src.utils.cache.pricing_cache.get_greeks", return_value=None)
    mocker.patch("src.pricing.factory.PricingEngineFactory.get_strategy", side_effect=Exception("Greeks fail"))
    
    payload = {
        "spot": 100, "strike": 100, "time_to_expiry": 1, 
        "volatility": 0.2, "rate": 0.05, "option_type": "call"
    }
    response = client.post("/pricing/greeks", json=payload)
    assert response.status_code == 422
