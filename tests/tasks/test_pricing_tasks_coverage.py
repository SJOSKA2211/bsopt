import pytest
from unittest.mock import MagicMock, patch
from src.tasks.pricing_tasks import price_option_task, batch_price_options_task, calculate_implied_volatility_task, generate_volatility_surface_task

@patch("src.tasks.pricing_tasks.PricingEngineFactory")
@patch("src.tasks.pricing_tasks.pricing_cache")
def test_price_option_task_success(mock_cache, mock_factory):
    # Mocking async methods
    async def mock_get(*args, **kwargs): return None
    async def mock_set(*args, **kwargs): return True
    
    mock_cache.get_option_price = MagicMock(side_effect=mock_get)
    mock_cache.set_option_price = MagicMock(side_effect=mock_set)
    
    mock_engine = MagicMock()
    mock_engine.price.return_value = 15.0
    mock_factory.get_strategy.return_value = mock_engine
    
    result = price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    
    assert result["price"] == 15.0
    # Since it's called inside a loop.run_until_complete, we check if the mock was called
    assert mock_cache.set_option_price.called

@patch("src.tasks.pricing_tasks.pricing_cache")
def test_price_option_task_cache_hit(mock_cache):
    async def mock_get(*args, **kwargs): return 20.0
    mock_cache.get_option_price = MagicMock(side_effect=mock_get)
    
    result = price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    
    assert result["price"] == 20.0
    assert result["cache_hit"] is True

@patch("src.tasks.pricing_tasks.PricingEngineFactory")
def test_batch_price_options_task_vectorized(mock_factory):
    # Mocking success case
    result = batch_price_options_task([{"spot": 100}], vectorized=True)
    assert "prices" in result
    assert result["vectorized"] is True

@patch("src.tasks.pricing_tasks.implied_volatility")
def test_calculate_implied_volatility_task(mock_iv):
    mock_iv.return_value = 0.25
    
    result = calculate_implied_volatility_task(price=10.0, spot=100, strike=100, maturity=1, rate=0.05, option_type="call")
    
    assert result["implied_vol"] == 0.25

def test_generate_volatility_surface_task():
    strikes = [90, 100, 110]
    maturities = [0.5, 1.0]
    prices = [[5, 10, 15], [6, 11, 16]]
    
    result = generate_volatility_surface_task(prices, strikes, maturities, spot=100, rate=0.05, option_type="call")
    
    assert "surface" in result
    assert len(result["surface"]) == 2

def test_price_option_task_invalid_params():
    with pytest.raises(ValueError, match="Invalid input parameters"):
        price_option_task(spot=-100, strike=100, maturity=1, volatility=0.2, rate=0.05)

def test_price_option_task_invalid_type():
    with pytest.raises(ValueError, match="Invalid option type"):
        price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="invalid")

@patch("src.tasks.pricing_tasks.pricing_cache")
def test_price_option_task_cache_exception(mock_cache):
    # Trigger exception in cache lookup
    mock_cache.get_option_price.side_effect = Exception("Cache down")
    
    # Should continue and succeed without cache
    result = price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, use_cache=True)
    assert "price" in result
    assert result["cache_hit"] is False

@patch("src.tasks.pricing_tasks.PricingEngineFactory")
@patch("src.tasks.pricing_tasks.pricing_cache")
def test_price_option_task_cache_set_exception(mock_cache, mock_factory):
    mock_cache.get_option_price.return_value = None
    mock_cache.set_option_price.side_effect = Exception("Cache set failed")
    
    mock_engine = MagicMock()
    mock_engine.price.return_value = 15.0
    mock_factory.get_strategy.return_value = mock_engine
    
    # Should succeed even if cache set fails
    result = price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert result["price"] == 15.0

@patch("src.tasks.pricing_tasks.PricingEngineFactory")
def test_price_option_task_general_exception(mock_factory):
    mock_factory.get_strategy.side_effect = Exception("General error")
    
    with pytest.raises(Exception, match="General error"):
        price_option_task(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)

def test_batch_price_options_task_non_vectorized():
    result = batch_price_options_task([{"spot": 100}], vectorized=False)
    assert len(result["prices"]) == 1
    assert result["vectorized"] is False