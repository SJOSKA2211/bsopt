import pytest
import numpy as np
from src.pricing.implied_vol import (
    implied_volatility, 
    vectorized_implied_volatility,
    ImpliedVolatilityError
)
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

def test_iv_single_newton():
    spot, strike, maturity, rate, vol = 100.0, 105.0, 0.5, 0.05, 0.25
    params = BSParameters(spot, strike, maturity, vol, rate)
    market_price = BlackScholesEngine.price_options(params=params, option_type="call")
    
    iv = implied_volatility(market_price, spot, strike, maturity, rate, method="newton")
    assert pytest.approx(iv, abs=1e-4) == vol

def test_iv_single_brent():
    spot, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.2
    params = BSParameters(spot, strike, maturity, vol, rate)
    market_price = BlackScholesEngine.price_options(params=params, option_type="call")
    
    iv = implied_volatility(market_price, spot, strike, maturity, rate, option_type="call", method="brent")
    assert pytest.approx(iv, abs=1e-4) == vol

def test_iv_invalid_inputs():
    with pytest.raises(ValueError, match="market_price cannot be negative"):
        implied_volatility(-1.0, 100.0, 100.0, 1.0, 0.05)
        
    with pytest.raises(ValueError, match="Arbitrage violation"):
        # Price below intrinsic
        implied_volatility(0.1, 100.0, 80.0, 1.0, 0.05, option_type="call")

def test_vectorized_iv():
    n = 10
    spots = np.full(n, 100.0)
    strikes = np.linspace(80.0, 120.0, n)
    vols = np.full(n, 0.2)
    maturities = np.full(n, 1.0)
    rates = np.full(n, 0.05)
    dividends = np.zeros(n)
    types = np.full(n, "call")
    
    # Generate market prices
    market_prices = BlackScholesEngine.price_options(
        spot=spots, strike=strikes, maturity=maturities,
        volatility=vols, rate=rates, dividend=dividends, option_type=types
    )
    
    calculated_vols = vectorized_implied_volatility(
        market_prices, spots, strikes, maturities, rates, dividends, types
    )
    
    assert not np.isnan(calculated_vols).any()
    assert np.allclose(calculated_vols, vols, atol=1e-3)
