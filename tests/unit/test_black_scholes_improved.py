import unittest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        self.params = BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.01
        )

    def test_price_options_scalar(self):
        engine = BlackScholesEngine()
        price = engine.price_options(params=self.params, option_type="call")
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)

    def test_price_options_vectorized(self):
        spots = np.array([100.0, 110.0])
        strikes = np.array([100.0, 100.0])
        engine = BlackScholesEngine()
        prices = engine.price_options(spot=spots, strike=strikes, maturity=1.0, volatility=0.2, rate=0.05)
        self.assertEqual(len(prices), 2)
        self.assertGreater(prices[1], prices[0])

    def test_calculate_greeks(self):
        engine = BlackScholesEngine()
        greeks = engine.calculate_greeks(params=self.params, option_type="call")
        self.assertIsInstance(greeks.delta, float)
        self.assertGreater(greeks.delta, 0)
        self.assertLess(greeks.delta, 1.0)

    def test_put_call_parity(self):
        engine = BlackScholesEngine()
        cp = engine.price_options(params=self.params, option_type="call")
        pp = engine.price_options(params=self.params, option_type="put")
        parity = engine.verify_put_call_parity(
            self.params.spot, self.params.strike, self.params.maturity, 
            self.params.rate, cp, pp, self.params.dividend
        )
        self.assertTrue(parity)

    def test_price_call_put(self):
        engine = BlackScholesEngine()
        call_p = engine.price_call(self.params)
        put_p = engine.price_put(self.params)
        self.assertGreater(call_p, 0)
        self.assertGreater(put_p, 0)

    def test_price_batch(self):
        engine = BlackScholesEngine()
        S = np.array([100.0, 100.0])
        K = np.array([100.0, 110.0])
        T = np.array([1.0, 1.0])
        sigma = np.array([0.2, 0.2])
        r = np.array([0.05, 0.05])
        q = np.array([0.0, 0.0])
        option_types = np.array(["call", "call"])
        prices = engine.price_batch(S, K, T, sigma, r, q, option_types)
        self.assertEqual(len(prices), 2)

    def test_calculate_greeks_batch(self):
        engine = BlackScholesEngine()
        S = np.array([100.0, 100.0])
        K = np.array([100.0, 110.0])
        T = np.array([1.0, 1.0])
        sigma = np.array([0.2, 0.2])
        r = np.array([0.05, 0.05])
        q = np.array([0.0, 0.0])
        greeks = engine.calculate_greeks_batch(spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q)
        self.assertIn("delta", greeks)
        self.assertEqual(len(greeks["delta"]), 2)

    def test_instance_price(self):
        engine = BlackScholesEngine()
        price = engine.price(self.params, option_type="call")
        self.assertGreater(price, 0)

    def test_module_level_funcs(self):
        from src.pricing.black_scholes import black_scholes, verify_put_call_parity as vpcp
        # When called with kwargs, returns float directly
        res = black_scholes(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
        self.assertIsInstance(res, float)
        
        # Test parity func
        parity = vpcp(self.params)
        self.assertTrue(parity)

if __name__ == '__main__':
    unittest.main()
