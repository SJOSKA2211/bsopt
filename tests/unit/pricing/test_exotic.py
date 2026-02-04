import pytest
import numpy as np
from src.pricing.exotic import (
    AsianOptionPricer, BarrierOptionPricer, LookbackOptionPricer, 
    DigitalOptionPricer, ExoticParameters, AsianType, BarrierType, StrikeType
)
from src.pricing.black_scholes import BSParameters

def test_geometric_asian():
    base = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    params = ExoticParameters(base_params=base, n_observations=252)
    
    price = AsianOptionPricer.price_geometric_asian(params, "call")
    assert price > 0
    # Asian price should be lower than European (usually)
    from src.pricing.black_scholes import BlackScholesEngine
    eur_price = BlackScholesEngine.price_options(params=base, option_type="call")
    assert price < eur_price

def test_barrier_analytical():
    base = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    # Down-and-out call: S=100, H=90
    params = ExoticParameters(base_params=base, barrier=90.0)
    
    price = BarrierOptionPricer.price_barrier_analytical(params, "call", BarrierType.DOWN_AND_OUT)
    assert price > 0
    
    # If barrier is breached at t=0, should handle it (here it raises ValueError if H >= S for down-out)
    with pytest.raises(ValueError):
        BarrierOptionPricer.price_barrier_analytical(params, "call", BarrierType.UP_AND_OUT)

def test_digital_option():
    base = BSParameters(spot=105.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    price = DigitalOptionPricer.price_cash_or_nothing(base, "call", payout=10.0)
    assert price > 0
    assert price < 10.0 # Discounted probability

def test_lookback_floating_strike():
    base = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    price = LookbackOptionPricer.price_floating_strike_analytical(base, "call")
    assert price > 0
