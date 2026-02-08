import unittest
import numpy as np
from src.pricing.implied_vol import implied_volatility, vectorized_implied_volatility, ImpliedVolatilityError

class TestImpliedVol(unittest.TestCase):
    def test_implied_vol_scalar(self):
        # S=100, K=100, T=1, r=0.05, vol=0.2 => Price approx 10.45
        price = 10.45
        iv = implied_volatility(price, 100.0, 100.0, 1.0, 0.05)
        self.assertAlmostEqual(iv, 0.2, delta=0.01)

    def test_implied_vol_vectorized(self):
        prices = np.array([10.45, 5.0])
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 110.0])
        maturities = np.array([1.0, 1.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        option_types = np.array(["call", "call"])
        
        ivs = vectorized_implied_volatility(prices, spots, strikes, maturities, rates, dividends, option_types)
        self.assertEqual(len(ivs), 2)
        self.assertTrue(np.all(ivs > 0))

    def test_arbitrage_violation(self):
        # Price below intrinsic
        with self.assertRaises(ValueError):
            implied_volatility(1.0, 100.0, 100.0, 1.0, 0.05)

if __name__ == '__main__':
    unittest.main()
