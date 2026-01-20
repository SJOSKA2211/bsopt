import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.tasks.pricing_tasks import price_option_task, batch_price_options_task, calculate_implied_volatility_task, generate_volatility_surface_task
from src.pricing.black_scholes import BSParameters

class MockTask:
    def __init__(self):
        self.request = MagicMock()
        self.request.id = "test-task-id"

@pytest.fixture
def mock_task():
    return MockTask()

@patch("src.tasks.pricing_tasks.pricing_cache", create=True)
@patch("src.tasks.pricing_tasks.asyncio")
def test_price_option_task_success(mock_asyncio, mock_cache, mock_task):
    # Setup cache miss
    mock_asyncio.get_event_loop().run_until_complete.return_value = None
    mock_asyncio.new_event_loop().run_until_complete.return_value = None
    
    result = price_option_task(
        mock_task,
        spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05,
        use_cache=True
    )
    
    assert result["status"] == "completed"
    assert result["price"] > 0
    assert result["cache_hit"] is False

@patch("src.tasks.pricing_tasks.pricing_cache", create=True)
@patch("src.tasks.pricing_tasks.asyncio")
def test_price_option_task_cache_hit(mock_asyncio, mock_cache, mock_task):
    # Setup cache hit
    # pricing_cache.get_option_price returns price
    # pricing_cache.get_greeks returns greeks object
    
    mock_greeks = MagicMock()
    mock_greeks.delta = 0.5
    mock_greeks.gamma = 0.1
    mock_greeks.vega = 0.2
    mock_greeks.theta = -0.1
    mock_greeks.rho = 0.1
    
    # We need to handle multiple calls to run_until_complete
    mock_asyncio.get_event_loop().run_until_complete.side_effect = [10.5, mock_greeks]
    
    result = price_option_task(
        mock_task,
        spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05,
        use_cache=True
    )
    
    assert result["status"] == "completed"
    assert result["price"] == 10.5
    assert result["cache_hit"] is True

def test_batch_price_options_task_vectorized(mock_task):
    options = [
        {"spot": 100, "strike": 100, "maturity": 1, "volatility": 0.2, "rate": 0.05},
        {"spot": 100, "strike": 110, "maturity": 1, "volatility": 0.2, "rate": 0.05}
    ] * 6 # Make it > 10 to trigger vectorized path
    
    result = batch_price_options_task(mock_task, options, use_vectorized=True)
    
    assert result["count"] == 12
    assert result["vectorized"] is True
    assert len(result["results"]) == 12
    assert result["results"][0]["price"] > 0

def test_calculate_implied_volatility_task(mock_task):
    # Price for vol=0.2 is approx 10.45
    market_price = 10.45
    
    result = calculate_implied_volatility_task(
        mock_task,
        market_price=market_price,
        spot=100, strike=100, maturity=1, rate=0.05,
        initial_guess=0.5
    )
    
    assert result["converged"] is True
    assert abs(result["implied_volatility"] - 0.2) < 0.05

def test_generate_volatility_surface_task(mock_task):
    strikes = [90, 100, 110]
    maturities = [0.5, 1.0, 2.0]
    
    result = generate_volatility_surface_task(
        mock_task,
        spot=100, strikes=strikes, maturities=maturities, rate=0.05
    )
    
    assert result["status"] == "completed"
    surface = result["surface"]
    assert len(surface) == len(maturities)
    assert len(surface[0]) == len(strikes)
