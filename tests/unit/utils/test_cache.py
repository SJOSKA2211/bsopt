import pytest
import json
import numpy as np
from src.utils.cache import (
    init_redis_cache, 
    close_redis_cache, 
    generate_cache_key, 
    pricing_cache, 
    rate_limiter, 
    idempotency_manager, 
    db_cache, 
    publish_to_redis, 
    get_redis_client,
    RateLimitTier,
    warm_cache
)
from src.pricing.models import BSParameters, OptionGreeks
from dataclasses import asdict

@pytest.mark.asyncio
async def test_redis_init_failure(mocker):
    # Mock ConnectionPool to fail
    mocker.patch("src.utils.cache.ConnectionPool.from_url", side_effect=Exception("Redis down"))
    res = await init_redis_cache()
    assert res is None

@pytest.mark.asyncio
async def test_generate_cache_key():
    params = {"spot": 100.0, "strike": 100.0}
    key1 = generate_cache_key("test", **params)
    # Ensure float64 vs float consistency
    key2 = generate_cache_key("test", spot=np.float64(100.0), strike=100.0)
    assert key1 == key2
    
    # Test numpy array
    key_arr = generate_cache_key("test", val=np.array([1, 2]))
    assert key_arr is not None
    
    with pytest.raises(TypeError):
        generate_cache_key("test", val=set())

@pytest.mark.asyncio
async def test_pricing_cache_redis_none(mocker):
    mocker.patch("src.utils.cache.get_redis", return_value=None)
    params = BSParameters(100, 100, 1, 0.2, 0.05)
    assert await pricing_cache.get_option_price(params, "call", "bs") is None
    assert await pricing_cache.set_option_price(params, "call", "bs", 10.0) is False
    assert await pricing_cache.get_greeks(params, "call") is None
    assert await pricing_cache.set_greeks(params, "call", OptionGreeks(0,0,0,0,0)) is False

@pytest.mark.asyncio
async def test_pricing_cache_redis_error(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    mock_redis.get = mocker.AsyncMock(side_effect=Exception("Redis error"))
    mock_redis.setex = mocker.AsyncMock(side_effect=Exception("Redis error"))
    
    params = BSParameters(100, 100, 1, 0.2, 0.05)
    assert await pricing_cache.get_option_price(params, "call", "bs") is None
    assert await pricing_cache.set_option_price(params, "call", "bs", 10.0) is False
    assert await pricing_cache.get_greeks(params, "call") is None
    assert await pricing_cache.set_greeks(params, "call", OptionGreeks(0,0,0,0,0)) is False

@pytest.mark.asyncio
async def test_rate_limiter_redis_none(mocker):
    mocker.patch("src.utils.cache.get_redis", return_value=None)
    assert await rate_limiter.check_rate_limit("u1", "/p") is True

@pytest.mark.asyncio
async def test_rate_limiter_error(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    mock_redis.pipeline.side_effect = Exception("Pipe fail")
    assert await rate_limiter.check_rate_limit("u1", "/p") is True

@pytest.mark.asyncio
async def test_pricing_cache(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    
    params = BSParameters(100, 100, 1, 0.2, 0.05)
    
    # set_option_price
    mock_redis.setex = mocker.AsyncMock()
    await pricing_cache.set_option_price(params, "call", "bs", 10.0)
    assert mock_redis.setex.called
    
    # get_option_price
    mock_redis.get = mocker.AsyncMock(return_value=json.dumps(10.0))
    price = await pricing_cache.get_option_price(params, "call", "bs")
    assert price == 10.0
    
    # get_greeks / set_greeks
    greeks = OptionGreeks(0.5, 0.1, -0.05, 0.2, 0.1)
    await pricing_cache.set_greeks(params, "call", greeks)
    mock_redis.get = mocker.AsyncMock(return_value=json.dumps(asdict(greeks)))
    cached_greeks = await pricing_cache.get_greeks(params, "call")
    assert cached_greeks.delta == 0.5

@pytest.mark.asyncio
async def test_rate_limiter(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    
    mock_pipe = mocker.Mock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_pipe.incr = mocker.Mock()
    mock_pipe.expire = mocker.Mock()
    mock_pipe.execute = mocker.AsyncMock(return_value=[1]) # Counter = 1
    
    res = await rate_limiter.check_rate_limit("user1", "/price", RateLimitTier.FREE)
    assert res is True

@pytest.mark.asyncio
async def test_idempotency_manager(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    
    mock_redis.set = mocker.AsyncMock(return_value=True)
    assert await idempotency_manager.check_and_set("key1") is True
    
    mock_redis.set = mocker.AsyncMock(return_value=False)
    assert await idempotency_manager.check_and_set("key1") is False

@pytest.mark.asyncio
async def test_db_cache(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    
    # set_user
    mock_redis.setex = mocker.AsyncMock(return_value=True)
    await db_cache.set_user("u1", {"name": "test"})
    assert mock_redis.setex.called
    
    # get_user
    mock_redis.get = mocker.AsyncMock(return_value=json.dumps({"name": "test"}))
    user = await db_cache.get_user("u1")
    assert user["name"] == "test"

@pytest.mark.asyncio
async def test_publish_to_redis(mocker):
    mock_redis = mocker.Mock()
    mocker.patch("src.utils.cache.get_redis", return_value=mock_redis)
    mock_redis.publish = mocker.AsyncMock()
    
    await publish_to_redis("chan1", {"msg": "hello"})
    assert mock_redis.publish.called

@pytest.mark.asyncio
async def test_get_redis_client_error(mocker):
    mocker.patch("src.utils.cache.get_redis", return_value=None)
    with pytest.raises(Exception): # FastAPI HTTPException
        await get_redis_client()

@pytest.mark.asyncio
async def test_warm_cache(mocker):
    mocker.patch("src.utils.cache.pricing_cache.set_option_price", mocker.AsyncMock())
    await warm_cache()
    # If no crash, it works

@pytest.mark.asyncio
async def test_close_redis_cache(mocker):
    mock_redis = mocker.Mock()
    mock_redis.close = mocker.AsyncMock()
    import src.utils.cache
    src.utils.cache._redis_pool = mock_redis
    await close_redis_cache()
    assert mock_redis.close.called
    assert src.utils.cache._redis_pool is None
