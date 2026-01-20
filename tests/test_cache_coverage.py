import json
import pytest
import importlib
from unittest.mock import MagicMock, AsyncMock, patch
import src.utils.cache
from src.utils.cache import (
    generate_cache_key,
    PricingCache,
    RateLimiter,
    RateLimitTier,
    BSParameters,
    OptionGreeks,
    init_redis_cache,
    close_redis_cache,
    warm_cache,
    idempotency_manager,
    db_cache,
    publish_to_redis,
    get_redis_client
)

@pytest.mark.asyncio
async def test_generate_cache_key():
    key = generate_cache_key("test", a=1, b=2.5, c=[1, 2])
    assert key.startswith("test:")
    assert len(key) > 10
    
    with pytest.raises(TypeError):
        generate_cache_key("test", d=object())

@pytest.mark.asyncio
async def test_pricing_cache_ops(monkeypatch):
    mock_redis = AsyncMock()
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    
    pc = PricingCache()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    # Set price
    await pc.set_option_price(params, "call", "bs", 10.5)
    mock_redis.setex.assert_called()
    
    # Get price
    mock_redis.get.return_value = json.dumps(10.5)
    price = await pc.get_option_price(params, "call", "bs")
    assert price == 10.5
    
    # Get greeks
    greeks = OptionGreeks(0.5, 0.01, -5.0, 10.0, 20.0)
    mock_redis.get.return_value = json.dumps({"delta": 0.5, "gamma": 0.01, "theta": -5.0, "vega": 10.0, "rho": 20.0})
    res_greeks = await pc.get_greeks(params, "call")
    assert res_greeks.delta == 0.5
    
    # Set greeks
    await pc.set_greeks(params, "call", greeks)
    mock_redis.setex.assert_called()

@pytest.mark.asyncio
async def test_rate_limiter(monkeypatch):
    mock_redis = MagicMock()
    mock_pipe = AsyncMock()
    mock_pipe.execute.return_value = [1]
    mock_redis.pipeline.return_value = mock_pipe
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    
    rl = RateLimiter()
    allowed = await rl.check_rate_limit("user1", "price", RateLimitTier.FREE)
    assert allowed is True

@pytest.mark.asyncio
async def test_init_close_redis(monkeypatch):
    # Mock redis and connection pool
    mock_pool = MagicMock()
    mock_redis = AsyncMock()
    monkeypatch.setattr("src.utils.cache.ConnectionPool.from_url", lambda *args, **kwargs: mock_pool)
    monkeypatch.setattr("src.utils.cache.Redis", MagicMock(return_value=mock_redis))
    
    # We need to bypass aioredis is None check if it was not installed (but it is)
    
    r = await init_redis_cache()
    assert r is not None
    await close_redis_cache()
    assert src.utils.cache._redis_pool is None

@pytest.mark.asyncio
async def test_warm_cache(monkeypatch):
    mock_redis = AsyncMock()
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    await warm_cache()
    assert mock_redis.setex.called

@pytest.mark.asyncio
async def test_idempotency(monkeypatch):
    mock_redis = AsyncMock()
    mock_redis.set.return_value = True
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    
    res = await idempotency_manager.check_and_set("key1")
    assert res is True

@pytest.mark.asyncio
async def test_db_cache(monkeypatch):
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps({"name": "user"})
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    
    user = await db_cache.get_user("123")
    assert user["name"] == "user"
    
    await db_cache.set_user("123", {"name": "user"})
    assert mock_redis.setex.called

@pytest.mark.asyncio
async def test_publish_to_redis(monkeypatch):
    mock_redis = AsyncMock()
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    await publish_to_redis("chan", {"msg": "hi"})
    assert mock_redis.publish.called

@pytest.mark.asyncio
async def test_get_redis_client(monkeypatch):
    mock_redis = MagicMock()
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    r = await get_redis_client()
    assert r == mock_redis
    
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: None)
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await get_redis_client()
