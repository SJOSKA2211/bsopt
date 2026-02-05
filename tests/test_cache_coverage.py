import importlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.utils.cache
from src.utils.cache import (
    BSParameters,
    OptionGreeks,
    PricingCache,
    RateLimiter,
    RateLimitTier,
    close_redis_cache,
    db_cache,
    generate_cache_key,
    get_redis_client,
    idempotency_manager,
    init_redis_cache,
    publish_to_redis,
    warm_cache,
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

@pytest.mark.asyncio
async def test_cache_no_redis(monkeypatch):
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: None)
    pc = PricingCache()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    assert await pc.get_option_price(params, "call", "bs") is None
    assert await pc.set_option_price(params, "call", "bs", 10.0) is False
    assert await pc.get_greeks(params, "call") is None
    assert await pc.set_greeks(params, "call", OptionGreeks(0,0,0,0,0)) is False
    
    rl = RateLimiter()
    assert await rl.check_rate_limit("u1", "p") is True
    
    assert await idempotency_manager.check_and_set("k") is True
    assert await db_cache.get_user("u") is None
    assert await db_cache.set_user("u", {}) is False

def test_redis_error_is_class():
    from src.utils.cache import RedisError
    assert issubclass(RedisError, Exception)

@pytest.mark.asyncio
async def test_cache_errors(monkeypatch):
    from src.utils.cache import RedisError
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = RedisError("Redis error")
    mock_redis.setex.side_effect = RedisError("Redis error")
    mock_redis.publish.side_effect = RedisError("Redis error")
    mock_pipe = AsyncMock()
    mock_pipe.execute.side_effect = RedisError("Pipe error")
    mock_redis.pipeline.return_value = mock_pipe
    
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    pc = PricingCache()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    assert await pc.get_option_price(params, "call", "bs") is None
    assert await pc.set_option_price(params, "call", "bs", 10.0) is False
    assert await pc.get_greeks(params, "call") is None
    assert await pc.set_greeks(params, "call", OptionGreeks(0,0,0,0,0)) is False
    
    rl = RateLimiter()
    assert await rl.check_rate_limit("u1", "p") is True
    
    assert await db_cache.get_user("u") is None
    assert await db_cache.set_user("u", {}) is False
    
    # test publish_to_redis error
    await publish_to_redis("chan", {"msg": "hi"}) # should log error but not raise

@pytest.mark.asyncio
async def test_get_redis_direct(monkeypatch):
    import src.utils.cache
    # Revert the autouse mock from conftest.py if it exists
    # We need the original function. Since it's a module level function, 
    # we can try to find it in the module's __dict__ if it wasn't overwritten yet,
    # but conftest.py overwrites it.
    
    # Let's define what get_redis SHOULD do and assert it matches the implementation
    # Actually, the best way is to reload the module or just not mock it in the first place.
    # But since we are here, let's just test it by calling the implementation directly 
    # if we can get a handle to it.
    
    # If we can't get the original, we can at least cover the lines by 
    # ensuring we don't mock it in a separate test file or by 
    # using a reload trick.
    importlib.reload(src.utils.cache)
    assert src.utils.cache.get_redis() is src.utils.cache._redis_pool

@pytest.mark.asyncio
async def test_init_redis_exception(monkeypatch):
    # This should trigger the except block in init_redis_cache
    monkeypatch.setattr("src.utils.cache.ConnectionPool.from_url", MagicMock(side_effect=RuntimeError("Init failed")))
    r = await init_redis_cache()
    assert r is None

@pytest.mark.asyncio
async def test_generate_cache_key_numpy():
    import numpy as np
    # Test np.integer and np.floating
    key1 = generate_cache_key("np", val=np.int64(10), fval=np.float64(1.23))
    assert key1.startswith("np:")
    
    # Test np.ndarray
    key2 = generate_cache_key("np", arr=np.array([1, 2, 3]))
    assert key2.startswith("np:")
    
    # Test unsupported type
    with pytest.raises(TypeError, match="not serializable"):
        generate_cache_key("np", val=set([1, 2]))

@pytest.mark.asyncio
async def test_pricing_cache_miss(monkeypatch):
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)
    
    pc = PricingCache()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    assert await pc.get_option_price(params, "call", "bs") is None
    assert await pc.get_greeks(params, "call") is None

@pytest.mark.asyncio
async def test_redis_error_type_hit(monkeypatch):
    import sys
    import types
    from unittest.mock import MagicMock

    import src.utils.cache
    
    # Create a mock module that looks real
    mock_aioredis = types.ModuleType("redis.asyncio")
    class RealError(Exception): pass
    mock_aioredis.RedisError = RealError
    mock_aioredis.Redis = MagicMock()
    mock_aioredis.ConnectionPool = MagicMock()
    
    # Modify sys.modules so reload sees it
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "redis.asyncio", mock_aioredis)
        importlib.reload(src.utils.cache)
        assert src.utils.cache.RedisError is RealError

@pytest.mark.asyncio
async def test_init_redis_no_aioredis_direct(monkeypatch):
    
    # Simulate aioredis import failure
    monkeypatch.setattr("src.utils.cache.aioredis", None)
    # We don't need to reload, just call it if aioredis is checked
    from src.utils.cache import init_redis_cache
    r = await init_redis_cache()
    assert r is None
