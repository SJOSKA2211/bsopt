import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.tasks.pricing_tasks import (
    price_option_task,
    batch_price_options_task,
    generate_volatility_surface_task
)

def test_vectorized_black_scholes_logic():
    from src.pricing.black_scholes import BlackScholesEngine, BSParameters
    spots = np.array([100.0, 110.0])
    strikes = np.array([100.0, 100.0])
    maturities = np.array([1.0, 1.0])
    volatilities = np.array([0.2, 0.2])
    rates = np.array([0.05, 0.05])
    dividends = np.array([0.02, 0.02])
    option_types = ["call", "put"]
    
    params = BSParameters(spots, strikes, maturities, volatilities, rates, dividends)
    prices = BlackScholesEngine.price_options(
        params=params, 
        option_type=option_types
    )
    
    assert len(prices) == 2
    assert not np.isnan(prices).any()
    assert prices[0] > 0

def test_price_option_task_no_cache():
    mock_self = MagicMock()
    mock_self.request.id = "test-task-id"
    
    # Signature: (self, spot, strike, maturity, volatility, rate, dividend=0.0, option_type='call', use_cache=True)
    # If _orig_run is the function itself, and it has 'self' as first arg, and it's bound...
    # Let's try calling it via the task object itself as positional
    result = price_option_task(
        100.0, 105.0, 0.5, 0.2, 0.05, 0.02, "call", False
    )
    
    assert result["status"] == "completed"

def test_price_option_task_invalid_input():
    with pytest.raises(ValueError, match="Invalid input parameters"):
        price_option_task(
            -100.0, 105.0, 0.5, 0.2, 0.05, 0.02, "call", False
        )

@patch("src.tasks.pricing_tasks.price_option_task.apply")
def test_batch_price_options_task_small(mock_apply):
    mock_self = MagicMock()
    mock_self.request.id = "batch-id"
    
    mock_res = MagicMock()
    mock_res.get.return_value = {"price": 10.0, "status": "completed", "task_id": "sub-id"}
    mock_apply.return_value = mock_res
    
    options = [
        {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05}
    ]
    
    # Signature: (self, options, use_vectorized=True)
    result = batch_price_options_task(options, False)
    
    assert result["count"] == 1
    assert result["results"][0]["price"] == 10.0

def test_generate_volatility_surface_task():
    strikes = [90.0, 100.0, 110.0]
    maturities = [0.25, 0.5, 1.0]
    
    # Signature: (self, spot, strikes, maturities, rate, dividend=0.0, base_volatility=0.2)
    result = generate_volatility_surface_task(100.0, strikes, maturities, 0.05, 0.02, 0.2)
    
    assert result["status"] == "completed"
    assert len(result["surface"]) == 3
    assert len(result["surface"][0]) == 3
