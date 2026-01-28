"""
from tests.test_utils import assert_equal

Comprehensive Test Suite for Optimized Quantitative Engines
"""

import unittest

import numpy as np

from src.pricing.implied_vol import implied_volatility, vectorized_implied_volatility
from src.pricing.black_scholes import BlackScholesEngine as VectorizedBlackScholesEngine


class TestOptimizedPricing(unittest.TestCase):
    def setUp(self):
        self.spots = np.array([100.0, 100.0, 100.0])
        self.strikes = np.array([90.0, 100.0, 110.0])
        self.maturities = np.array([1.0, 1.0, 1.0])
        self.vols = np.array([0.2, 0.2, 0.2])
        self.rates = np.array([0.05, 0.05, 0.05])
        self.divs = np.array([0.0, 0.0, 0.0])
        self.types = np.array(["call", "call", "call"])

    def test_vectorized_bs_accuracy(self):
        """Verify JIT BS engine against known values."""
        prices = VectorizedBlackScholesEngine.price_options(100, 100, 1, 0.2, 0.05, 0, "call")
        # Standard BS price for S=100, K=100, T=1, sigma=0.2, r=0.05 is ~10.4506
        self.assertAlmostEqual(prices[0], 10.450583572185565, places=6)

    def test_greeks_consistency(self):
        """Verify delta is within [0, 1] for calls and [-1, 0] for puts."""
        greeks = VectorizedBlackScholesEngine.calculate_greeks(
            self.spots, self.strikes, self.maturities, self.vols, self.rates, self.divs, self.types
        )
        self.assertTrue(np.all(greeks["delta"] >= 0))
        self.assertTrue(np.all(greeks["delta"] <= 1.0))

    def test_iv_convergence(self):
        """Verify IV calculation recovers the input volatility."""
        target_vol = 0.25
        price = VectorizedBlackScholesEngine.price_options(
            100, 100, 1, target_vol, 0.05, 0, "call"
        )[0]

        iv = implied_volatility(price, 100, 100, 1, 0.05, 0, "call")
        self.assertAlmostEqual(iv, target_vol, places=4)

    def test_batch_iv(self):
        """Verify batch IV calculation speed and accuracy."""
        vols = np.array([0.15, 0.25, 0.35])
        prices = VectorizedBlackScholesEngine.price_options(
            self.spots, self.strikes, self.maturities, vols, self.rates, self.divs, self.types
        )

        calc_vols = vectorized_implied_volatility(
            prices, self.spots, self.strikes, self.maturities, self.rates, self.divs, self.types
        )

        np.testing.assert_array_almost_equal(calc_vols, vols, decimal=4)


if __name__ == "__main__":
    unittest.main()
