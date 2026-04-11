import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

import pytest
from fastapi.testclient import TestClient

# Mock the cache decorator BEFORE importing app/routes
with patch("src.shared.utils.cache.multi_layer_cache", lambda *args, **kwargs: (lambda f: f)):
    from api.index import app
    from api.routes.pricing import pricing_service

from src.auth.auth import get_current_active_user, get_current_user, auth_service
from src.auth.core.tokens import TokenData
from src.database.models import User
from src.shared.utils.cache import get_redis_client
from api.schemas.pricing import PriceResult, OptionGreeksStruct, BatchPriceResult

def create_mock_redis():
    mock_redis = AsyncMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = AsyncMock(return_value=[None, 0]) # Mock L2 cache miss
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex = AsyncMock(return_value=True)
    return mock_redis

GLOBAL_MOCK_REDIS = create_mock_redis()

@pytest.fixture
def mock_user():
    return User(
        id="test-user-id",
        email="test@example.com",
        tier="pro",
        is_active=True,
        is_verified=True
    )

@pytest.fixture(autouse=True)
def setup_api_mocks(mock_user, monkeypatch):
    """Mock authentication and rate limiting for each test."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    app.dependency_overrides[get_redis_client] = lambda: GLOBAL_MOCK_REDIS
    
    # Mock get_redis globally
    monkeypatch.setattr("src.shared.utils.cache.get_redis", lambda: GLOBAL_MOCK_REDIS)
    
    from api.middleware.jwt_validator import require_auth
    mock_claims = MagicMock()
    mock_claims.tier = "pro"
    mock_claims.user_id = "test-user-id"
    app.dependency_overrides[require_auth] = lambda: mock_claims

    # Patch the middleware's auth service hop
    with patch.object(auth_service, "validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = TokenData(
            user_id="test-user-id",
            email="test@example.com",
            tier="pro",
            token_type="access",
            jti="jti123",
            exp=time.time() + 3600,
            iat=time.time()
        )
        yield mock_val
        
    app.dependency_overrides.clear()

client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture(autouse=True)
def reset_circuits():
    from src.shared.utils.circuit_breaker import pricing_circuit
    pricing_circuit.reset()
    yield

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    # pricing_service calls engine.price_european
    result = MagicMock()
    result.price = 10.5
    result.greeks = MagicMock(delta=0.5, gamma=0.05, theta=-0.01, vega=0.1, rho=0.02)
    engine.price_european.return_value = result
    
    # for calculate_greeks
    from api.schemas.pricing import OptionGreeksStruct
    engine.calculate_greeks.return_value = OptionGreeksStruct(
        delta=0.5, gamma=0.05, theta=-0.01, vega=0.1, rho=0.02
    )
    return engine

def test_calculate_price_success(mock_engine):
    with patch("src.math_kernel.service.PricingEngineFactory.get_engine", return_value=mock_engine):
        payload = {
            "spot": 100.0,
            "strike": 105.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes",
        }
        response = client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["price"] == 10.5

def test_calculate_price_invalid_params(mock_engine):
    # PricingService.price_option catches generic Exception and raises 500
    mock_engine.price_european.side_effect = Exception("Invalid spot price")
    with patch("src.math_kernel.service.PricingEngineFactory.get_engine", return_value=mock_engine):
        payload = {
            "spot": -100.0, # Invalid
            "strike": 105.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes",
        }
        response = client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 500

def test_calculate_price_validation_error():
    payload = {"strike": 105.0} # Missing required fields
    response = client.post(
        "/api/v1/pricing/price", 
        json=payload,
        headers={"Authorization": "Bearer some_token"}
    )
    assert response.status_code == 422

def test_calculate_batch_prices_success():
    with patch.object(pricing_service, "price_batch", new_callable=AsyncMock) as mock_batch:
        mock_batch.return_value = BatchPriceResult(
            results=[],
            total_count=0,
            computation_time_ms=10.0,
            cached_count=0
        )
        payload = {
            "options": [
                {
                    "spot": 100.0,
                    "strike": 105.0,
                    "time_to_expiry": 0.5,
                    "rate": 0.05,
                    "volatility": 0.2,
                    "option_type": "call",
                    "model": "black_scholes",
                }
            ]
        }
        response = client.post(
            "/api/v1/pricing/batch", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200

def test_calculate_greeks_success(mock_engine):
    with patch("src.math_kernel.service.PricingEngineFactory.get_engine", return_value=mock_engine):
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
        }
        response = client.post(
            "/api/v1/pricing/greeks", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200

def test_calculate_price_heston_success():
    payload = {
        "spot": 100.0,
        "strike": 105.0,
        "time_to_expiry": 0.5,
        "rate": 0.05,
        "volatility": 0.2,
        "option_type": "call",
        "model": "heston",
        "symbol": "SPY",
    }
    
    with patch.object(pricing_service, "price_option", new_callable=AsyncMock) as mock_price:
        mock_price.return_value = PriceResult(
            price=12.34,
            spot=100.0,
            strike=105.0,
            time_to_expiry=0.5,
            rate=0.05,
            volatility=0.2,
            option_type="call",
            model="heston",
            computation_time_ms=1.0,
            cached=False,
            greeks=None
        )
        response = client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert response.json()["price"] == 12.34
