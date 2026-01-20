import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock, AsyncMock
import os
import json
from src.security.auth import get_current_user_flexible
from src.utils.cache import get_redis_client
from src.security.rate_limit import rate_limit

client = TestClient(app)

@pytest.fixture(autouse=True)
def set_testing_env():
    with patch.dict(os.environ, {"TESTING": "true"}):
        yield

@pytest.fixture(autouse=True)
def override_dependencies():
    async def mock_get_current_user():
        return {"sub": "user123"}
    
    mock_redis = MagicMock()
    # Mock pipeline to be awaitable
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=(1, None))
    mock_redis.pipeline.return_value = mock_pipe
    mock_redis.get = AsyncMock(return_value=None)
    
    async def mock_get_redis():
        return mock_redis

    async def mock_rate_limit():
        return None

    app.dependency_overrides[get_current_user_flexible] = mock_get_current_user
    app.dependency_overrides[get_redis_client] = mock_get_redis
    app.dependency_overrides[rate_limit] = mock_rate_limit
    yield mock_redis
    app.dependency_overrides.clear()

VALID_PRICE_REQUEST = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_expiry": 1.0,
    "volatility": 0.2,
    "rate": 0.05,
    "dividend_yield": 0.0,
    "option_type": "call",
    "model": "black_scholes"
}

def test_calculate_price_success(override_dependencies):
    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):
        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.price.return_value = 10.45
            mock_factory.return_value = mock_strategy
            
            response = client.post("/api/v1/pricing/price", json=VALID_PRICE_REQUEST)
            assert response.status_code == 200
            assert response.json()["data"]["price"] == 10.45

def test_calculate_price_cached(override_dependencies):
    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=10.45):
        response = client.post("/api/v1/pricing/price", json=VALID_PRICE_REQUEST)
        assert response.status_code == 200
        assert response.json()["data"]["price"] == 10.45
        assert response.json()["message"] == "Result retrieved from cache"

def test_calculate_price_heston_success(override_dependencies):
    mock_redis = override_dependencies
    heston_request = VALID_PRICE_REQUEST.copy()
    heston_request["model"] = "heston"
    heston_request["symbol"] = "AAPL"
    
    # Mocking redis_client.get
    mock_redis.get = AsyncMock(return_value=json.dumps({
        "params": {
            "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7
        },
        "timestamp": 1000000000000 # Very fresh
    }))
    
    with patch('time.time', return_value=1000000000000 + 100):
        with patch('src.api.routes.pricing.HestonModelFFT') as mock_heston:
            mock_model = MagicMock()
            mock_model.price_call.return_value = 12.34
            mock_heston.return_value = mock_model
            
            response = client.post("/api/v1/pricing/price", json=heston_request)
            assert response.status_code == 200
            assert response.json()["data"]["price"] == 12.34
            assert response.headers["X-Pricing-Model"] == "Heston-FFT"

def test_calculate_price_heston_missing_symbol(override_dependencies):
    heston_request = VALID_PRICE_REQUEST.copy()
    heston_request["model"] = "heston"
    # No symbol
    
    response = client.post("/api/v1/pricing/price", json=heston_request)
    assert response.status_code == 422
    assert "Symbol is required" in response.json()["message"]

def test_calculate_price_heston_stale_fallback(override_dependencies):
    mock_redis = override_dependencies
    heston_request = VALID_PRICE_REQUEST.copy()
    heston_request["model"] = "heston"
    heston_request["symbol"] = "AAPL"
    
    mock_redis.get = AsyncMock(return_value=json.dumps({
        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7},
        "timestamp": 1000000000 # Very old
    }))
    
    with patch('time.time', return_value=2000000000):
        with patch('src.utils.cache.pricing_cache.get_option_price', return_value=10.45):
            response = client.post("/api/v1/pricing/price", json=heston_request)
            assert response.status_code == 200
            assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"

def test_calculate_batch_prices(override_dependencies):
    batch_request = {
        "options": [VALID_PRICE_REQUEST, VALID_PRICE_REQUEST]
    }
    
    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):
        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.price.return_value = 10.45
            mock_factory.return_value = mock_strategy
            
            response = client.post("/api/v1/pricing/batch", json=batch_request)
            assert response.status_code == 200
            assert len(response.json()["data"]["results"]) == 2

def test_calculate_greeks(override_dependencies):
    greeks_request = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "volatility": 0.2,
        "rate": 0.05,
        "dividend_yield": 0.0,
        "option_type": "call"
    }
    
    with patch('src.utils.cache.pricing_cache.get_greeks', return_value=None):
        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:
            mock_strategy = MagicMock()
            mock_greeks = MagicMock()
            mock_greeks.delta = 0.6
            mock_greeks.gamma = 0.02
            mock_greeks.theta = -0.01
            mock_greeks.vega = 0.1
            mock_greeks.rho = 0.05
            mock_strategy.calculate_greeks.return_value = mock_greeks
            mock_strategy.price.return_value = 10.45
            mock_factory.return_value = mock_strategy
            
            response = client.post("/api/v1/pricing/greeks", json=greeks_request)
            assert response.status_code == 200
            assert response.json()["data"]["delta"] == 0.6

def test_calculate_iv(override_dependencies):
    iv_request = {
        "option_price": 10.45,
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "rate": 0.05,
        "dividend_yield": 0.0,
        "option_type": "call"
    }
    
    with patch('src.api.routes.pricing.implied_volatility', return_value=0.2):
        response = client.post("/api/v1/pricing/implied-volatility", json=iv_request)
        assert response.status_code == 200
        assert response.json()["data"]["implied_volatility"] == 0.2

def test_calculate_price_heston_put(override_dependencies):

    mock_redis = override_dependencies

    heston_request = VALID_PRICE_REQUEST.copy()

    heston_request["model"] = "heston"

    heston_request["symbol"] = "AAPL"

    heston_request["option_type"] = "put"

    

    mock_redis.get = AsyncMock(return_value=json.dumps({

        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7},

        "timestamp": 1000000000000

    }))

    

    with patch('time.time', return_value=1000000000000 + 100):

        with patch('src.api.routes.pricing.HestonModelFFT') as mock_heston:

            mock_model = MagicMock()

            mock_model.price_put.return_value = 5.67

            mock_heston.return_value = mock_model

            

            response = client.post("/api/v1/pricing/price", json=heston_request)

            assert response.status_code == 200

            assert response.json()["data"]["price"] == 5.67



def test_calculate_price_heston_error_fallback(override_dependencies):

    mock_redis = override_dependencies

    heston_request = VALID_PRICE_REQUEST.copy()

    heston_request["model"] = "heston"

    heston_request["symbol"] = "AAPL"

    

    mock_redis.get = AsyncMock(return_value=json.dumps({

        "params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7},

        "timestamp": 1000000000000

    }))

    

    with patch('time.time', return_value=1000000000000 + 100):

        with patch('src.api.routes.pricing.HestonModelFFT', side_effect=Exception("Heston crash")):

            with patch('src.utils.cache.pricing_cache.get_option_price', return_value=10.45):

                response = client.post("/api/v1/pricing/price", json=heston_request)

                assert response.status_code == 200

                assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"



def test_calculate_price_value_error(override_dependencies):

    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):

        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:

            mock_strategy = MagicMock()

            mock_strategy.price.side_effect = ValueError("Invalid sigma")

            mock_factory.return_value = mock_strategy

            

            response = client.post("/api/v1/pricing/price", json=VALID_PRICE_REQUEST)

            assert response.status_code == 422

            assert "Invalid parameters" in response.json()["message"]



def test_calculate_price_unexpected_error(override_dependencies):

    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):

        with patch('src.pricing.factory.PricingEngineFactory.get_strategy', side_effect=Exception("Unknown error")):

            response = client.post("/api/v1/pricing/price", json=VALID_PRICE_REQUEST)

            assert response.status_code == 500

            assert "Internal error" in response.json()["message"]



def test_calculate_batch_prices_with_error(override_dependencies):

    batch_request = {

        "options": [VALID_PRICE_REQUEST, VALID_PRICE_REQUEST]

    }

    

    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):

        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:

            mock_strategy = MagicMock()

            # First one succeeds, second one fails

            mock_strategy.price.side_effect = [10.45, Exception("Calculation failed")]

            mock_factory.return_value = mock_strategy

            

            response = client.post("/api/v1/pricing/batch", json=batch_request)

            assert response.status_code == 200

            # Should have only 1 result because second one failed and was skipped

            assert len(response.json()["data"]["results"]) == 1



def test_calculate_greeks_error(override_dependencies):

    greeks_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call"

    }

    with patch('src.utils.cache.pricing_cache.get_greeks', return_value=None):

        with patch('src.pricing.factory.PricingEngineFactory.get_strategy', side_effect=Exception("Greeks failed")):

            response = client.post("/api/v1/pricing/greeks", json=greeks_request)

            assert response.status_code == 422

            assert "Greeks calculation failed" in response.json()["message"]



def test_calculate_iv_value_error(override_dependencies):

    iv_request = {

        "option_price": 10.45, "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call"

    }

    with patch('src.api.routes.pricing.implied_volatility', side_effect=ValueError("No convergence")):

        response = client.post("/api/v1/pricing/implied-volatility", json=iv_request)

        assert response.status_code == 422

        assert "IV calculation failed" in response.json()["message"]



def test_calculate_iv_unexpected_error(override_dependencies):

    iv_request = {

        "option_price": 10.45, "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call"

    }

    with patch('src.api.routes.pricing.implied_volatility', side_effect=Exception("Unexpected")):

        response = client.post("/api/v1/pricing/implied-volatility", json=iv_request)

        assert response.status_code == 500

        assert "Internal error" in response.json()["message"]



def test_calculate_exotic_barrier_missing_type(override_dependencies):

    exotic_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "barrier", "barrier": 120.0

    }

    response = client.post("/api/v1/pricing/exotic", json=exotic_request)

    assert response.status_code == 422

    assert "barrier_type required" in response.json()["message"]



def test_calculate_exotic_invalid_type(override_dependencies):

    exotic_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "barrier", "barrier_type": "invalid", "barrier": 120.0

    }

    response = client.post("/api/v1/pricing/exotic", json=exotic_request)

    assert response.status_code == 422

    assert "Invalid barrier_type" in response.json()["message"]



def test_calculate_exotic_asian_lookback_digital(override_dependencies):

    # Asian

    asian_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "asian", "asian_type": "arithmetic"

    }

    with patch('src.api.routes.pricing.price_exotic_option', return_value=(9.5, None)):

        response = client.post("/api/v1/pricing/exotic", json=asian_request)

        assert response.status_code == 200

        

    # Lookback

    lookback_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "lookback"

    }

    with patch('src.api.routes.pricing.price_exotic_option', return_value=(15.0, None)):

        response = client.post("/api/v1/pricing/exotic", json=lookback_request)

        assert response.status_code == 200



    # Digital

    digital_request = {

        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,

        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "digital", "payout": 100.0

    }

    with patch('src.api.routes.pricing.price_exotic_option', return_value=(60.0, None)):

        response = client.post("/api/v1/pricing/exotic", json=digital_request)

        assert response.status_code == 200



def test_calculate_exotic_error(override_dependencies):



    exotic_request = {



        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,



        "dividend_yield": 0.0, "option_type": "call", "exotic_type": "asian"



    }



    with patch('src.api.routes.pricing.price_exotic_option', side_effect=Exception("Exotic fail")):



        response = client.post("/api/v1/pricing/exotic", json=exotic_request)



        assert response.status_code == 422



        assert "Pricing failed" in response.json()["message"]







def test_calculate_price_circuit_breaker(override_dependencies):



    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=None):



        with patch('src.pricing.factory.PricingEngineFactory.get_strategy', side_effect=Exception("Circuit Breaker open")):



            response = client.post("/api/v1/pricing/price", json=VALID_PRICE_REQUEST)



            assert response.status_code == 503



            assert "Circuit Breaker" in response.json()["message"]







def test_calculate_batch_prices_cached(override_dependencies):



    batch_request = {



        "options": [VALID_PRICE_REQUEST]



    }



    with patch('src.utils.cache.pricing_cache.get_option_price', return_value=10.45):



        response = client.post("/api/v1/pricing/batch", json=batch_request)



        assert response.status_code == 200



        assert response.json()["data"]["cached_count"] == 1







def test_calculate_greeks_cached(override_dependencies):



    greeks_request = {



        "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, "volatility": 0.2, "rate": 0.05,



        "dividend_yield": 0.0, "option_type": "call"



    }



    mock_greeks = MagicMock()



    mock_greeks.delta = 0.6



    mock_greeks.gamma = 0.02



    mock_greeks.theta = -0.01



    mock_greeks.vega = 0.1



    mock_greeks.rho = 0.05



    



    with patch('src.utils.cache.pricing_cache.get_greeks', return_value=mock_greeks):



        with patch('src.pricing.factory.PricingEngineFactory.get_strategy') as mock_factory:



            mock_strategy = MagicMock()



            mock_strategy.price.return_value = 10.45



            mock_factory.return_value = mock_strategy



            



            response = client.post("/api/v1/pricing/greeks", json=greeks_request)



            assert response.status_code == 200



            assert response.json()["data"]["delta"] == 0.6








