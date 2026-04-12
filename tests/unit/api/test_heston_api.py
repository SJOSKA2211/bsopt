import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from api.index import app
from src.auth.auth import get_current_active_user
from src.shared.utils.cache import get_redis_client
from src.auth.rate_limit import rate_limit

@pytest.fixture
def mock_user():
    from src.database.models import User
    from uuid import uuid4
    return User(
        id=str(uuid4()),
        email="test@example.com",
        tier="pro",
        is_active=True,
        is_verified=True
    )

@pytest.fixture
def heston_client(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    app.dependency_overrides[rate_limit] = lambda: None
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

class TestPricingAPIHeston:
    @pytest.mark.asyncio
    async def test_heston_pricing_success(self, heston_client, mocker):
        """Verify Heston pricing when parameters are available in Redis."""
        # 1. Setup mock Redis data
        mock_params = {
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7,
        }
        import time
        import json

        mock_cache = {
            "params": mock_params,
            "timestamp": time.time(),
            "metrics": {"rmse": 0.01},
        }

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(mock_cache))
        
        # Mock pipeline for multi_layer_cache decorator
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(return_value=[None, 0])  # [cached_val, remaining_ms]
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        # Patch get_redis in the cache module
        mocker.patch("src.shared.utils.cache.get_redis", return_value=mock_redis)

        # 2. Make request
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.03,
            "volatility": 0.2,
            "option_type": "call",
            "model": "heston",
            "symbol": "SPY",
        }

        response = heston_client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer test-token"}
        )

        # 3. Verify
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "heston"
        assert response.headers["X-Pricing-Model"] == "Heston-FFT"
        assert data["price"] > 0

    @pytest.mark.asyncio
    async def test_heston_fallback_to_bs(self, heston_client, mocker):
        """Verify fallback to Black-Scholes when Redis is empty."""
        # 1. Setup mock Redis (Empty)
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        
        # Mock pipeline for multi_layer_cache decorator
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(return_value=[None, 0])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        mocker.patch("src.shared.utils.cache.get_redis", return_value=mock_redis)

        # 2. Make request
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.03,
            "volatility": 0.2,
            "option_type": "call",
            "model": "heston",
            "symbol": "SPY",
        }

        response = heston_client.post(
            "/api/v1/pricing/price", 
            json=payload,
            headers={"Authorization": "Bearer test-token"}
        )

        # 3. Verify
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "black_scholes"
        assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"
