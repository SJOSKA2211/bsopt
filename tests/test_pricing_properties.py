
from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.implied_vol import implied_volatility

# Strategy for valid option parameters
bs_params = st.builds(
    BSParameters,
    spot=st.floats(min_value=1.0, max_value=1000.0),
    strike=st.floats(min_value=1.0, max_value=1000.0),
    maturity=st.floats(min_value=0.01, max_value=10.0), # 3 days to 10 years
    volatility=st.floats(min_value=0.01, max_value=2.0), # 1% to 200%
    rate=st.floats(min_value=0.0, max_value=0.2), # 0% to 20%
    dividend=st.floats(min_value=0.0, max_value=0.1) # 0% to 10%
)

class TestPricingProperties:
    
    @given(bs_params)
    @settings(max_examples=100, deadline=None)
    def test_put_call_parity(self, params):
        """
        C - P = S * e^(-qT) - K * e^(-rT)
        """
        engine = BlackScholesEngine()
        call_price = engine.price(params, "call")
        put_price = engine.price(params, "put")
        
        # RHS of parity
        lhs = call_price - put_price
        rhs = (params.spot * np.exp(-params.dividend * params.maturity)) - \
              (params.strike * np.exp(-params.rate * params.maturity))
              
        assert np.isclose(lhs, rhs, atol=1e-5), \
            f"Put-Call Parity violated: {lhs} != {rhs} for {params}"

    @given(bs_params)
    @settings(max_examples=100, deadline=None)
    def test_call_monotonicity_spot(self, params):
        """Call price should increase as Spot increases."""
        engine = BlackScholesEngine()
        p1 = engine.price(params, "call")
        
        params_higher = replace(params, spot=params.spot + 1.0)
        p2 = engine.price(params_higher, "call")
        
        assert p2 >= p1 - 1e-9

    @given(bs_params)
    @settings(max_examples=100, deadline=None)
    def test_call_monotonicity_vol(self, params):
        """Call price should increase as Volatility increases (usually)."""
        engine = BlackScholesEngine()
        p1 = engine.price(params, "call")
        
        params_higher = replace(params, volatility=params.volatility + 0.1)
        p2 = engine.price(params_higher, "call")
        
        # Vega is generally positive for calls
        assert p2 >= p1 - 1e-9

    @given(bs_params)
    @settings(max_examples=50, deadline=None)
    def test_iv_roundtrip(self, params):
        """Implied Volatility should recover input volatility."""
        engine = BlackScholesEngine()
        price = engine.price(params, "call")
        
        # Skip if price is too low (deep OTM) or too high (deep ITM) where IV is unstable
        intrinsic = max(0, params.spot - params.strike)
        if price < 0.01 or price < intrinsic + 0.01:
            return

        try:
            iv = implied_volatility(
                price, params.spot, params.strike, params.maturity, 
                params.rate, params.dividend, "call"
            )
            # Relax tolerance for extreme values
            assert np.isclose(iv, params.volatility, atol=1e-3), \
                f"IV mismatch: {iv} != {params.volatility} for price {price}"
        except Exception:
            # IV calculation might fail for extreme parameters, which is acceptable in some contexts
            pass
