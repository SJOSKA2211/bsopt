import pytest
import json
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from src.utils.cache import (
    init_redis_cache, close_redis_cache, generate_cache_key, 
    pricing_cache, rate_limiter, warm_cache, IdempotencyManager,
    RateLimitTier, RATE_LIMIT_CONFIGS
)
from src.pricing.models import BSParameters, OptionGreeks

@pytest.mark.asyncio
async def test_init_redis_cache_success():
    with patch("src.utils.cache.Redis") as mock_redis_class:
        mock_redis_instance = AsyncMock()
        mock_redis_class.return_value = mock_redis_instance
        
        res = await init_redis_cache()
        assert res is not None
        mock_redis_instance.ping.assert_called_once()

@pytest.mark.asyncio
async def test_init_redis_cache_failure():
    with patch("src.utils.cache.ConnectionPool.from_url") as mock_pool:
        mock_pool.side_effect = Exception("Connection failed")
        res = await init_redis_cache()
        assert res is None

def test_generate_cache_key():
    key1 = generate_cache_key("prefix", a=1, b=2)
    key2 = generate_cache_key("prefix", b=2, a=1)
    assert key1 == key2
    
    # Test with numpy types
    key3 = generate_cache_key("prefix", val=np.float64(1.5))
    assert "prefix:" in key3

@pytest.mark.asyncio
async def test_pricing_cache_get_set():
    with patch("src.utils.cache.get_redis") as mock_get_redis:
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        
        params = BSParameters(100, 100, 1, 0.2, 0.05)
        
        # Set
        await pricing_cache.set_option_price(params, "call", "bs", 10.5)
        mock_redis.setex.assert_called()
        
        # Get
        mock_redis.get.return_value = b"10.5"
        val = await pricing_cache.get_option_price(params, "call", "bs")
        assert val == 10.5

@pytest.mark.asyncio
async def test_pricing_cache_greeks():
    with patch("src.utils.cache.get_redis") as mock_get_redis:
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        
        params = BSParameters(100, 100, 1, 0.2, 0.05)
        greeks = OptionGreeks(0.5, 0.1, -0.05, 0.2, 0.1)
        
        await pricing_cache.set_greeks(params, "call", greeks)
        mock_redis.setex.assert_called()
        
        mock_redis.get.return_value = json.dumps({"delta": 0.5, "gamma": 0.1, "theta": -0.05, "vega": 0.2, "rho": 0.1}).encode()
        cached_greeks = await pricing_cache.get_greeks(params, "call")
        assert cached_greeks.delta == 0.5

@pytest.mark.asyncio
async def test_rate_limiter():
    with patch("src.utils.cache.get_redis") as mock_get_redis:
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        
        # Mock pipeline
        mock_pipe = MagicMock() 
        mock_pipe.execute = AsyncMock(return_value=[1])
        # Ensure calling pipeline() returns the mock_pipe object, not a coroutine
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        
        allowed = await rate_limiter.check_rate_limit("user1", "pricing")
        assert allowed is True

        
        mock_pipe.execute.return_value = [1000] # Over limit
        allowed = await rate_limiter.check_rate_limit("user1", "pricing", tier=RateLimitTier.FREE)
        assert allowed is False


@pytest.mark.asyncio
async def test_warm_cache():
    with patch("src.utils.cache.pricing_cache.set_option_price") as mock_set:
        await warm_cache()
        assert mock_set.called

@pytest.mark.asyncio
async def test_close_redis_cache():
    with patch("src.utils.cache._redis_pool", AsyncMock()) as mock_pool:
        await close_redis_cache()
        mock_pool.close.assert_called_once()

def test_generate_cache_key_complex():
    # Test with array (line 106)
    key = generate_cache_key("test", arr=np.array([1, 2, 3]))
    assert "test:" in key
    
    # Test with unknown type (line 108)
    with pytest.raises(TypeError, match="not serializable"):
        generate_cache_key("test", val=object())

@pytest.mark.asyncio
async def test_pricing_cache_redis_none():
    with patch("src.utils.cache.get_redis", return_value=None):
        assert await pricing_cache.get_option_price(None, "call", "bs") is None
        assert await pricing_cache.set_option_price(None, "call", "bs", 10.0) is False
        assert await pricing_cache.get_greeks(None, "call") is None
        assert await pricing_cache.set_greeks(None, "call", None) is False

@pytest.mark.asyncio
async def test_rate_limiter_redis_none():
    with patch("src.utils.cache.get_redis", return_value=None):
        assert await rate_limiter.check_rate_limit("user", "endpoint") is True

@pytest.mark.asyncio
async def test_rate_limiter_exception():
    with patch("src.utils.cache.get_redis") as mock_get:
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Redis error")
        mock_get.return_value = mock_redis
        assert await rate_limiter.check_rate_limit("user", "endpoint") is True

@pytest.mark.asyncio
async def test_idempotency_manager_full():
    with patch("src.utils.cache.get_redis") as mock_get:
        mock_redis = AsyncMock()
        mock_get.return_value = mock_redis
        
        mgr = IdempotencyManager()
        
        # check_and_set
        mock_redis.set.return_value = True
        assert await mgr.check_and_set("key") is True
        
        mock_redis.set.return_value = False
        assert await mgr.check_and_set("key") is False


@pytest.mark.asyncio
async def test_database_query_cache():
    from src.utils.cache import db_cache
    with patch("src.utils.cache.get_redis") as mock_get:
        mock_redis = AsyncMock()
        mock_get.return_value = mock_redis
        
        # set_user
        await db_cache.set_user("u1", {"name": "test"})
        mock_redis.setex.assert_called()
        
        # get_user
        mock_redis.get.return_value = json.dumps({"name": "test"}).encode()
        assert await db_cache.get_user("u1") == {"name": "test"}

@pytest.mark.asyncio
async def test_publish_to_redis():
    from src.utils.cache import publish_to_redis
    with patch("src.utils.cache.get_redis") as mock_get:
        mock_redis = AsyncMock()
        mock_get.return_value = mock_redis
        
        await publish_to_redis("chan1", {"msg": "hi"})
        mock_redis.publish.assert_called_once_with("chan1", json.dumps({"msg": "hi"}))

@pytest.mark.asyncio
async def test_get_redis_client():
    from src.utils.cache import get_redis_client
    with patch("src.utils.cache.get_redis", return_value=None):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await get_redis_client()
    
    with patch("src.utils.cache.get_redis", return_value=MagicMock()):
        assert await get_redis_client() is not None



