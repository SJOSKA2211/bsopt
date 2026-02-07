import json
from unittest.mock import AsyncMock, MagicMock


class TestPricingAPIHeston:
    def test_heston_pricing_success(self, api_client):
        """Verify Heston pricing when parameters are available in Redis."""
        # 1. Setup mock Redis data
        mock_params = {
            "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7
        }
        import time
        mock_cache = {"params": mock_params, "timestamp": time.time(), "metrics": {"rmse": 0.01}}
        
        from src.utils.cache import get_redis_client
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(mock_cache))
        
        # Override dependencies
        from src.api.main import app
        from src.security.auth import get_current_user_flexible
        from src.security.rate_limit import rate_limit
        
        app.dependency_overrides[get_redis_client] = lambda: mock_redis
        app.dependency_overrides[get_current_user_flexible] = lambda: {"id": "test-user", "tier": "free"}
        app.dependency_overrides[rate_limit] = lambda: None
        
        # 2. Make request
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.03,
            "volatility": 0.2,
            "option_type": "call",
            "model": "heston",
            "symbol": "SPY"
        }
        
        response = api_client.post("/api/v1/pricing/price", json=payload)
        
        # 3. Verify
        assert response.status_code == 200
        data = response.json()['data']
        assert data['model'] == "heston"
        assert response.headers["X-Pricing-Model"] == "Heston-FFT"
        assert data['price'] > 0

    def test_heston_fallback_to_bs(self, api_client):
        """Verify fallback to Black-Scholes when Redis is empty."""
        # 1. Setup mock Redis (Empty)
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        
        from src.api.main import app
        from src.security.auth import get_current_user_flexible
        from src.security.rate_limit import rate_limit
        from src.utils.cache import get_redis_client
        
        app.dependency_overrides[get_redis_client] = lambda: mock_redis
        app.dependency_overrides[get_current_user_flexible] = lambda: {"id": "test-user", "tier": "free"}
        app.dependency_overrides[rate_limit] = lambda: None
        
        # 2. Make request
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.03,
            "volatility": 0.2,
            "option_type": "call",
            "model": "heston",
            "symbol": "SPY"
        }
        
        response = api_client.post("/api/v1/pricing/price", json=payload)
        
        # 3. Verify
        assert response.status_code == 200
        data = response.json()['data']
        assert data['model'] == "black_scholes"
        assert response.headers["X-Pricing-Model"] == "Black-Scholes-Fallback"