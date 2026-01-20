import pytest
import numpy as np
import math
from src.pricing.implied_vol import (
    implied_volatility,
    vectorized_implied_volatility,
    _calculate_intrinsic_value,
    _validate_inputs,
    _newton_raphson_iv,
    _brent_iv,
    ImpliedVolatilityError
)
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

def test_calculate_intrinsic_value():
    # Call ITM: 110 - 100 = 10
    assert math.isclose(_calculate_intrinsic_value(110, 100, 0.0, 0.0, 1.0, "call"), 10.0)
    # Put ITM: 100 - 90 = 10
    assert math.isclose(_calculate_intrinsic_value(90, 100, 0.0, 0.0, 1.0, "put"), 10.0)
    # OTM
    assert _calculate_intrinsic_value(90, 100, 0.0, 0.0, 1.0, "call") == 0.0

def test_validate_inputs():
    with pytest.raises(ValueError, match="market_price cannot be negative"):
        _validate_inputs(-1, 100, 100, 1, 0.05, 0, "call")
    with pytest.raises(ValueError, match="spot must be positive"):
        _validate_inputs(10, 0, 100, 1, 0.05, 0, "call")
    with pytest.raises(ValueError, match="strike must be positive"):
        _validate_inputs(10, 100, 0, 1, 0.05, 0, "call")
    with pytest.raises(ValueError, match="maturity must be positive"):
        _validate_inputs(10, 100, 100, 0, 0.05, 0, "call")
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        _validate_inputs(10, 100, 100, 1, 0.05, 0, "invalid")
    
    # Arbitrage violation
    with pytest.raises(ValueError, match="Arbitrage violation"):
        _validate_inputs(5, 110, 100, 1, 0.0, 0.0, "call") # Intrinsic is 10
        
    # Zero market price - Use r=0, q=0 so intrinsic is 0
    with pytest.raises(ImpliedVolatilityError, match="market price too close to zero"):
        _validate_inputs(1e-13, 100, 100, 1, 0.0, 0.0, "call")

def test_vectorized_implied_volatility_early_break():
    # Force early convergence
    spots = np.array([100.0])
    strikes = np.array([100.0])
    maturities = np.array([1.0])
    rates = np.array([0.05])
    dividends = np.array([0.0])
    option_types = np.array(["call"])
    # Get price for 20% vol
    market_price = BlackScholesEngine.price_call(BSParameters(100, 100, 1, 0.2, 0.05))
    # Give exact price and high tolerance
    ivs = vectorized_implied_volatility(np.array([market_price]), spots, strikes, maturities, rates, dividends, option_types, tolerance=1.0)
    # It should break at first iteration
    assert len(ivs) == 1

def test_newton_raphson_iv_validation():
    with pytest.raises(ValueError, match="initial_guess must be positive"):
        _newton_raphson_iv(10, 100, 100, 1, 0.05, 0, "call", initial_guess=0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        _newton_raphson_iv(10, 100, 100, 1, 0.05, 0, "call", tolerance=0)
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        _newton_raphson_iv(10, 100, 100, 1, 0.05, 0, "call", max_iterations=0)

def test_implied_volatility_methods():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    market_price = BlackScholesEngine.price_call(params)
    
    # Auto
    iv_auto = implied_volatility(market_price, 100, 100, 1.0, 0.05)
    assert math.isclose(iv_auto, 0.2, rel_tol=1e-5)
    
    # Newton
    iv_newton = implied_volatility(market_price, 100, 100, 1.0, 0.05, method="newton")
    assert math.isclose(iv_newton, 0.2, rel_tol=1e-5)
    
    # Brent
    iv_brent = implied_volatility(market_price, 100, 100, 1.0, 0.05, method="brent")
    assert math.isclose(iv_brent, 0.2, rel_tol=1e-5)
    
    with pytest.raises(ValueError, match="method must be 'auto', 'newton', or 'brent'"):
        implied_volatility(market_price, 100, 100, 1.0, 0.05, method="invalid")

def test_implied_volatility_failure_fallback():
    # Force Newton to fail by setting max_iterations=1 and a far guess
    # Newton should fail, and fallback to Brent in 'auto' mode
    iv = implied_volatility(10.0, 100, 100, 1.0, 0.05, method="auto", max_iterations=1, initial_guess=0.01)
    assert iv > 0
    
    # Newton failure without fallback
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        implied_volatility(10.0, 100, 100, 1.0, 0.05, method="newton", max_iterations=1, initial_guess=0.01)

def test_brent_iv_failure(mocker):
    mocker.patch("scipy.optimize.brentq", side_effect=Exception("Brent failed"))
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        _brent_iv(10, 100, 100, 1, 0.05, 0, "call")

def test_vectorized_implied_volatility():
    spots = np.array([100.0, 100.0])
    strikes = np.array([100.0, 110.0])
    maturities = np.array([1.0, 1.0])
    rates = np.array([0.05, 0.05])
    dividends = np.array([0.0, 0.0])
    option_types = np.array(["call", "call"])
    market_prices = np.array([10.45, 5.50]) # Approx prices for 20% vol
    
    ivs = vectorized_implied_volatility(market_prices, spots, strikes, maturities, rates, dividends, option_types)
    assert len(ivs) == 2
    assert np.all(ivs > 0)
    assert np.all(np.isfinite(ivs))

def test_vectorized_implied_volatility_non_convergence():
    # Use very low iterations to force non-convergence
    spots = np.array([100.0])
    strikes = np.array([100.0])
    maturities = np.array([1.0])
    rates = np.array([0.05])
    dividends = np.array([0.0])
    option_types = np.array(["call"])
    market_prices = np.array([10.45])
    
    ivs = vectorized_implied_volatility(market_prices, spots, strikes, maturities, rates, dividends, option_types, max_iterations=0)
    assert np.isnan(ivs[0])

def test_vectorized_implied_volatility_empty():
    empty = np.array([])
    res = vectorized_implied_volatility(empty, empty, empty, empty, empty, empty, empty)
    assert len(res) == 0

def test_newton_raphson_iv_small_vega():
    # Deep OTM option has very small vega
    # market_price=0.0001, spot=10, strike=100
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        _newton_raphson_iv(0.0001, 10, 100, 1.0, 0.05, 0.0, "call")