import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app
from src.auth.auth import get_current_active_user, get_current_user, auth_service
from src.auth.core.tokens import TokenData
from src.database.models import User
from src.shared.utils.cache import get_redis_client


def create_mock_redis():
    mock_redis = AsyncMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = AsyncMock(return_value=[1, True])
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    return mock_redis


GLOBAL_MOCK_REDIS = create_mock_redis()


@pytest.fixture
def mock_user():
    return User(
        id=MagicMock(),
        email="test@example.com",
        tier="free",
        is_active=True,
        is_verified=True
    )


@pytest.fixture(autouse=True)
def setup_api_mocks(mock_user):
    """Mock authentication and rate limiting for each test."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    app.dependency_overrides[get_redis_client] = lambda: GLOBAL_MOCK_REDIS
    
    # Patch the middleware's auth service hop
    with patch.object(auth_service, "validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = TokenData(
            user_id="1",
            email="test@example.com",
            tier="free",
            token_type="access",
            jti="jti123",
            exp=datetime.now() if 'datetime' in globals() else time.time() + 3600,
            iat=time.time()
        )
        yield mock_val
        
    app.dependency_overrides.clear()


client = TestClient(app, raise_server_exceptions=False)

from api.middleware.jwt_validator import require_auth
@pytest.fixture(autouse=True)
def override_auth():
    mock_claims = MagicMock()
    mock_claims.tier = "pro"
    app.dependency_overrides[require_auth] = lambda: mock_claims
    yield
    app.dependency_overrides.clear()



@pytest.fixture(autouse=True)
def reset_circuits():
    from src.shared.utils.circuit_breaker import pricing_circuit
    pricing_circuit.reset()
    yield


@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.price.return_value = 10.5
    strategy.calculate_greeks.return_value = MagicMock(
        delta=0.5, gamma=0.05, theta=-0.01, vega=0.1, rho=0.02
    )
    return strategy


def test_calculate_price_success(mock_strategy):
    with patch("src.math_kernel.factory.PricingEngineFactory.get_engine", return_value=mock_strategy):
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


def test_calculate_price_invalid_params(mock_strategy):
    mock_strategy.price.side_effect = ValueError("Invalid spot price")
    with patch("src.math_kernel.factory.PricingEngineFactory.get_engine", return_value=mock_strategy):
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
        # Should be 400 or 422 depending on how ValueError is handled
        assert response.status_code in (400, 422, 500)


def test_calculate_price_validation_error():
    payload = {"strike": 105.0} # Missing required fields
    response = client.post(
        "/api/v1/pricing/price", 
        json=payload,
        headers={"Authorization": "Bearer some_token"}
    )
    assert response.status_code == 422


def test_calculate_batch_prices_success(mock_strategy):
    from api.routes.pricing import pricing_service
    with patch.object(pricing_service, "price_batch", new_callable=AsyncMock) as mock_batch:
        from api.schemas.pricing import BatchPriceResult
        mock_batch.return_value = BatchPriceResult(
            results=[],
            total_count=2,
            cached_count=0,
            computation_time_ms=10.0
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


def test_calculate_greeks_success(mock_strategy):
    from api.routes.pricing import pricing_service
    with patch.object(pricing_service, "calculate_greeks", new_callable=AsyncMock) as mock_greeks:
        mock_greeks.return_value = {"delta": 0.5, "gamma": 0.05}
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
    
    # Mock PricingService.price_option directly to avoid complex Heston setup
    from api.routes.pricing import pricing_service
    with patch.object(pricing_service, "price_option", new_callable=AsyncMock) as mock_price:
        from api.schemas.pricing import PriceResult
        mock_price.return_value = PriceResult(
            price=12.34,
            model="Heston-FFT",
            spot=100.0,
            strike=105.0,
            cached=False,
            greeks={}, computation_time_ms=0.0
        )
        response = client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert response.json()["price"] == 12.34
