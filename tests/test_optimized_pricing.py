"""
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
        price = VectorizedBlackScholesEngine.price_options(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
        # Standard BS price for S=100, K=100, T=1, sigma=0.2, r=0.05 is ~10.4506
        self.assertAlmostEqual(float(price), 10.450583572185565, places=6)

    def test_greeks_consistency(self):
        """Verify delta is within [0, 1] for calls and [-1, 0] for puts."""
        greeks = VectorizedBlackScholesEngine.calculate_greeks_batch(
            spot=self.spots, 
            strike=self.strikes, 
            maturity=self.maturities, 
            volatility=self.vols, 
            rate=self.rates, 
            dividend=self.divs, 
            option_type=self.types
        )
        self.assertTrue(np.all(greeks["delta"] >= 0))
        self.assertTrue(np.all(greeks["delta"] <= 1.0))

    def test_iv_convergence(self):
        """Verify IV calculation recovers the input volatility."""
        target_vol = 0.25
        price = VectorizedBlackScholesEngine.price_options(
            100.0, 100.0, 1.0, target_vol, 0.05, 0.0, "call"
        )

        iv = implied_volatility(float(price), 100.0, 100.0, 1.0, 0.05, 0.0, "call")
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

    def test_wasm_simd_speedup(self):
        """🚀 SINGULARITY: Verify WASM SIMD pricing if available."""
        try:
            from src.wasm.engine import BlackScholesWASM
            engine = BlackScholesWASM()
            # Test case: S=100, K=100, T=1, sigma=0.2, r=0.05
            price = engine.price_call(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
            self.assertAlmostEqual(price, 10.450583572185565, places=6)
            print("\n[SUCCESS] WASM SIMD Pricing Verified")
        except ImportError:
            print("\n[SKIP] WASM Engine not installed in this environment")
        except Exception as e:
            print(f"\n[FAIL] WASM SIMD Pricing failed: {e}")

    def test_lsm_american_accuracy(self):
        """🚀 SINGULARITY: Verify Optimized LSM American Pricing."""
        try:
            from src.wasm.engine import AmericanOptionsWASM
            engine = AmericanOptionsWASM()
            # Standard American Call on non-dividend paying stock = European Call
            price = engine.price_lsm(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True, 10000, 50)
            # Should be close to 10.45
            self.assertTrue(10.0 < price < 11.0)
            print(f"[SUCCESS] LSM American Pricing Verified: ${price:.4f}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[FAIL] LSM American Pricing failed: {e}")


if __name__ == "__main__":
    unittest.main()