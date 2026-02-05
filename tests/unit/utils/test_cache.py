from unittest.mock import AsyncMock, patch

import pytest

from src.pricing.black_scholes import BSParameters
from src.utils.cache import (
    IdempotencyManager,
    PricingCache,
    RateLimiter,
    RateLimitTier,
    generate_cache_key,
)


def test_generate_cache_key():
    key1 = generate_cache_key("test", spot=100.0, strike=100.0)
    key2 = generate_cache_key("test", strike=100.0, spot=100.0)
    assert key1 == key2
    assert "test:" in key1

@pytest.mark.asyncio
@patch("src.utils.cache.get_redis")
async def test_pricing_cache(mock_get_redis):
    mock_redis = AsyncMock()
    mock_get_redis.return_value = mock_redis
    
    cache = PricingCache()
    params = BSParameters(100.0, 100.0, 1.0, 0.2, 0.05)
    
    # Set
    await cache.set_option_price(params, "call", "bs", 10.5)
    assert mock_redis.setex.called
    
    # Get hit
    mock_redis.get.return_value = "10.5"
    price = await cache.get_option_price(params, "call", "bs")
    assert price == 10.5
    
    # Get miss
    mock_redis.get.return_value = None
    price = await cache.get_option_price(params, "call", "bs")
    assert price is None

@pytest.mark.asyncio
@patch("src.utils.cache.get_redis")
async def test_idempotency_manager(mock_get_redis):
    mock_redis = AsyncMock()
    mock_get_redis.return_value = mock_redis
    
    idem = IdempotencyManager()
    
    # First time
    mock_redis.set.return_value = True
    assert await idem.check_and_set("key1") is True
    
    # Second time
    mock_redis.set.return_value = False
    assert await idem.check_and_set("key1") is False

@pytest.mark.asyncio
@patch("src.utils.cache.get_redis")
async def test_rate_limiter(mock_get_redis):
    mock_redis = AsyncMock()
    mock_get_redis.return_value = mock_redis
    
    # Mock pipeline
    mock_pipe = AsyncMock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_pipe.execute.return_value = [1] # result of INCR
    
    limiter = RateLimiter()
    allowed = await limiter.check_rate_limit("user1", "/price", RateLimitTier.FREE)
    assert allowed is True
