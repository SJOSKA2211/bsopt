import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.models import BSParameters, OptionGreeks

class TestBlackScholesEngine:
    
    def test_scalar_call_price(self):
        # Known value: S=100, K=100, T=1, sigma=0.2, r=0.05, q=0.0
        # Call ~ 10.4506
        price = BlackScholesEngine.price_options(
            spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0, option_type="call"
        )
        assert np.isclose(price, 10.4506, atol=1e-4)

    def test_scalar_put_price(self):
        # Put ~ 5.5735
        price = BlackScholesEngine.price_options(
            spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0, option_type="put"
        )
        assert np.isclose(price, 5.5735, atol=1e-4)

    def test_vectorized_pricing(self):
        S = np.array([100.0, 100.0])
        K = np.array([100.0, 110.0])
        T = np.array([1.0, 1.0])
        sigma = np.array([0.2, 0.2])
        r = np.array([0.05, 0.05])
        q = np.array([0.0, 0.0])
        
        prices = BlackScholesEngine.price_options(
            spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q, option_type="call"
        )
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 2
        assert np.isclose(prices[0], 10.4506, atol=1e-4)
        assert prices[1] < prices[0]  # Higher strike call is cheaper

    def test_greeks_calculation(self):
        greeks = BlackScholesEngine.calculate_greeks(
            spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0, option_type="call"
        )
        # Check explicit OptionGreeks or dict return
        if isinstance(greeks, dict):
            assert "delta" in greeks
            assert 0.0 < greeks["delta"] < 1.0
        else:
            assert 0.0 < greeks.delta < 1.0

    def test_put_call_parity(self):
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
        call = BlackScholesEngine.price_options(S, K, T, 0.2, r, q, "call")
        put = BlackScholesEngine.price_options(S, K, T, 0.2, r, q, "put")
        
        is_valid = BlackScholesEngine.verify_put_call_parity(S, K, T, r, call, put, q)
        assert is_valid

    def test_missing_spot_raises_error(self):
        with pytest.raises(ValueError, match="Missing spot price"):
            BlackScholesEngine.price_options(strike=100.0)

    def test_params_object(self):
        # Mock BSParameters if needed, or assume it works as a named tuple/pydantic
        # Trying with a dummy object
        class MockParams:
            spot = 100.0
            strike = 100.0
            maturity = 1.0
            volatility = 0.2
            rate = 0.05
            dividend = 0.0
        
        price = BlackScholesEngine.price_options(params=MockParams(), option_type="call")
        assert np.isclose(price, 10.4506, atol=1e-4)

    def test_price_wrapper(self):
        """Test the instance method wrapper .price()"""
        engine = BlackScholesEngine()
        price = engine.price(
            spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0, option_type="call"
        )
        assert np.isclose(price, 10.4506, atol=1e-4)

    def test_mixed_option_types(self):
        """Test vectorized pricing with mixed 'call' and 'put' types."""
        S = np.array([100.0, 100.0])
        K = np.array([100.0, 100.0])
        T = np.array([1.0, 1.0])
        sigma = np.array([0.2, 0.2])
        r = np.array([0.05, 0.05])
        q = np.array([0.0, 0.0])
        types = np.array(["call", "put"])
        
        prices = BlackScholesEngine.price_options(
            spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q, option_type=types
        )
        assert len(prices) == 2
        assert np.isclose(prices[0], 10.4506, atol=1e-4) # Call
        assert np.isclose(prices[1], 5.5735, atol=1e-4)  # Put

    def test_greeks_mixed_types(self):
        """Test greeks with mixed option types."""
        S = np.array([100.0, 100.0])
        types = np.array(["call", "put"])
        
        greeks = BlackScholesEngine.calculate_greeks(
            spot=S, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0, option_type=types
        )
        assert len(greeks["delta"]) == 2
        assert 0.0 < greeks["delta"][0] < 1.0 # Call delta
        assert -1.0 < greeks["delta"][1] < 0.0 # Put delta
