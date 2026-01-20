import pytest
import numpy as np
import math
from src.pricing.exotic import (
    ExoticParameters,
    AsianType,
    BarrierType,
    StrikeType,
    AsianOptionPricer,
    BarrierOptionPricer,
    LookbackOptionPricer,
    DigitalOptionPricer,
    price_exotic_option
)
from src.pricing.black_scholes import BSParameters

@pytest.fixture
def base_params():
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)

@pytest.fixture
def exotic_params(base_params):
    return ExoticParameters(base_params)

def test_exotic_parameters(base_params):
    params = ExoticParameters(base_params, n_observations=100, barrier=120.0, rebate=5.0)
    assert params.n_observations == 100
    assert params.barrier == 120.0
    assert params.rebate == 5.0

def test_asian_geometric(exotic_params):
    price = AsianOptionPricer.price_geometric_asian(exotic_params, "call")
    assert price > 0
    price_put = AsianOptionPricer.price_geometric_asian(exotic_params, "put")
    assert price_put > 0
    
    # Zero maturity
    params_zero = ExoticParameters(BSParameters(100, 100, 0.0, 0.2, 0.05))
    assert AsianOptionPricer.price_geometric_asian(params_zero, "call") == 0.0

def test_asian_arithmetic_mc(exotic_params):
    price, ci = AsianOptionPricer.price_arithmetic_asian_mc(exotic_params, "call", n_paths=1000, seed=42)
    assert price > 0
    assert ci > 0
    
    # Floating strike
    price_fl, _ = AsianOptionPricer.price_arithmetic_asian_mc(exotic_params, "call", strike_type=StrikeType.FLOATING, n_paths=1000)
    assert price_fl > 0
    
    # Put
    price_put, _ = AsianOptionPricer.price_arithmetic_asian_mc(exotic_params, "put", n_paths=1000)
    assert price_put > 0

    # Zero maturity
    params_zero = ExoticParameters(BSParameters(100, 100, 0.0, 0.2, 0.05))
    p_z, _ = AsianOptionPricer.price_arithmetic_asian_mc(params_zero, "call")
    assert p_z == 0.0

def test_barrier_analytical(exotic_params):
    # Down and out
    exotic_params.barrier = 80.0
    price = BarrierOptionPricer.price_barrier_analytical(exotic_params, "call", BarrierType.DOWN_AND_OUT)
    assert price > 0
    
    # Up and out
    exotic_params.barrier = 120.0
    price_uo = BarrierOptionPricer.price_barrier_analytical(exotic_params, "call", BarrierType.UP_AND_OUT)
    assert price_uo > 0
    
    # Validation errors
    with pytest.raises(ValueError, match="Up-barrier must be above spot price"):
        BarrierOptionPricer.price_barrier_analytical(ExoticParameters(BSParameters(100, 100, 1, 0.2, 0.05), barrier=90), "call", BarrierType.UP_AND_OUT)
    with pytest.raises(ValueError, match="Down-barrier must be below spot price"):
        BarrierOptionPricer.price_barrier_analytical(ExoticParameters(BSParameters(100, 100, 1, 0.2, 0.05), barrier=110), "call", BarrierType.DOWN_AND_OUT)

def test_lookback_floating_analytical(base_params):
    price = LookbackOptionPricer.price_floating_strike_analytical(base_params, "call")
    assert price > 0
    price_put = LookbackOptionPricer.price_floating_strike_analytical(base_params, "put")
    assert price_put > 0
    
    # Zero maturity
    base_zero = BSParameters(100, 100, 0.0, 0.2, 0.05)
    assert LookbackOptionPricer.price_floating_strike_analytical(base_zero, "call") == 0.0

def test_lookback_mc(exotic_params):
    # Floating
    price, _ = LookbackOptionPricer.price_lookback_mc(exotic_params, "call", StrikeType.FLOATING, n_paths=1000)
    assert price > 0
    # Fixed
    price_fix, _ = LookbackOptionPricer.price_lookback_mc(exotic_params, "call", StrikeType.FIXED, n_paths=1000)
    assert price_fix > 0
    # Put
    price_put, _ = LookbackOptionPricer.price_lookback_mc(exotic_params, "put", StrikeType.FLOATING, n_paths=1000)
    assert price_put > 0
    price_put_fix, _ = LookbackOptionPricer.price_lookback_mc(exotic_params, "put", StrikeType.FIXED, n_paths=1000)
    assert price_put_fix > 0

def test_digital_pricer(base_params):
    price_cn = DigitalOptionPricer.price_cash_or_nothing(base_params, "call")
    assert price_cn > 0
    price_cn_put = DigitalOptionPricer.price_cash_or_nothing(base_params, "put")
    assert price_cn_put > 0
    
    price_an = DigitalOptionPricer.price_asset_or_nothing(base_params, "call")
    assert price_an > 0
    price_an_put = DigitalOptionPricer.price_asset_or_nothing(base_params, "put")
    assert price_an_put > 0
    
    greeks = DigitalOptionPricer.calculate_digital_greeks(base_params, "call")
    assert greeks.delta > 0
    greeks_put = DigitalOptionPricer.calculate_digital_greeks(base_params, "put")
    assert greeks_put.delta < 0

def test_lookback_extrema():
    paths = np.array([[100, 110, 90], [100, 95, 105]])
    indices = np.array([0, 1, 2])
    max_ext = LookbackOptionPricer._compute_running_extrema(paths, indices, mode="max")
    assert np.array_equal(max_ext, [110, 105])
    min_ext = LookbackOptionPricer._compute_running_extrema(paths, indices, mode="min")
    assert np.array_equal(min_ext, [90, 95])

def test_price_exotic_option_dispatch(exotic_params):
    # Asian
    p, _ = price_exotic_option("asian", exotic_params, "call", asian_type=AsianType.GEOMETRIC)
    assert p > 0
    p_mc, ci = price_exotic_option("asian", exotic_params, "call", asian_type=AsianType.ARITHMETIC, n_paths=100)
    assert p_mc > 0
    
    # Barrier with string
    p_b, _ = price_exotic_option("barrier", exotic_params, "call", barrier_type="down-and-out")
    assert p_b > 0
    
    # Barrier with Enum
    p_b_enum, _ = price_exotic_option("barrier", exotic_params, "call", barrier_type=BarrierType.DOWN_AND_OUT)
    assert p_b_enum > 0
    with pytest.raises(ValueError, match="Barrier type is required"):
        price_exotic_option("barrier", exotic_params, "call")
        
    # Lookback
    p_l, _ = price_exotic_option("lookback", exotic_params, "call", strike_type=StrikeType.FLOATING, use_mc=False)
    assert p_l > 0
    p_l_mc, _ = price_exotic_option("lookback", exotic_params, "call", strike_type=StrikeType.FIXED, use_mc=True)
    assert p_l_mc > 0
    
    # Digital
    p_d, _ = price_exotic_option("digital", exotic_params, "call", payout=10.0)
    assert p_d > 0
    
    # Unknown
    with pytest.raises(ValueError, match="Unknown exotic option type"):
        price_exotic_option("unknown", exotic_params, "call")

def test_barrier_in_type(exotic_params):
    # Hit the 'else' block in BarrierOptionPricer.price_barrier_analytical
    exotic_params.barrier = 80.0
    price = BarrierOptionPricer.price_barrier_analytical(exotic_params, "call", BarrierType.DOWN_AND_IN)
    assert price > 0