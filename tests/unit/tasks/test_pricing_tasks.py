import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.tasks.pricing_tasks import (
    price_option_task, 
    batch_price_options_task, 
    calculate_implied_volatility_task,
    generate_volatility_surface_task
)

def test_vectorized_black_scholes_logic():
    spots = [100.0, 110.0]
    strikes = [100.0, 100.0]
    maturities = [1.0, 1.0]
    vols = [0.2, 0.2]
    rates = [0.05, 0.05]
    
    prices = BlackScholesEngine.price_options(
        spot=spots, strike=strikes, maturity=maturities, volatility=vols, rate=rates
    )
    assert len(prices) == 2
    assert prices[0] > 0

def test_price_option_task_no_cache():
    result = price_option_task(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, use_cache=False
    )
    assert result["status"] == "completed"
    assert result["price"] > 0
    assert "delta" in result

def test_price_option_task_invalid_input():
    with pytest.raises(ValueError, match="all must be positive"):
        price_option_task(spot=-100, strike=100, maturity=1, volatility=0.2, rate=0.05)

def test_batch_price_options_task_small():
    options = [
        {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05},
        {"spot": 110.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05}
    ]
    # Small batch uses apply() on individual tasks
    result = batch_price_options_task(options, use_vectorized=False)
    assert result["count"] == 2
    assert len(result["results"]) == 2

def test_generate_volatility_surface_task():
    strikes = [90.0, 100.0, 110.0]
    maturities = [0.5, 1.0]
    result = generate_volatility_surface_task(
        spot=100.0, strikes=strikes, maturities=maturities, rate=0.05
    )
    assert result["status"] == "completed"
    assert len(result["surface"]) == 2
    assert len(result["surface"][0]) == 3

def test_calculate_implied_volatility_task():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    market_price = BlackScholesEngine.price_call(params)
    
    result = calculate_implied_volatility_task(
        market_price=market_price, spot=100, strike=100, maturity=1.0, rate=0.05
    )
    assert result["status"] == "completed"
    assert result["converged"] is True
    assert np.isclose(result["implied_volatility"], 0.2, atol=1e-4)

def test_calculate_implied_volatility_task_failure():
    with pytest.raises(ValueError, match="Vega too small"):
        calculate_implied_volatility_task(
            market_price=1000.0, spot=100, strike=100, maturity=1e-30, rate=0.05
        )