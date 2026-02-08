import unittest
import numpy as np
from src.pricing.models.heston_fft import HestonModelFFT, batch_heston_price_jit, HestonParams

class TestHestonFFT(unittest.TestCase):
    def setUp(self):
        # 2κθ > σ² => 2 * 2 * 0.04 > 0.1² => 0.16 > 0.01 (Valid)
        self.params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)
        self.r = 0.05
        self.T = 1.0

    def test_price_surface_fft(self):
        model = HestonModelFFT(self.params, self.r, self.T)
        prices = model.price_surface_fft(S0=100.0, K_min=80.0, K_max=120.0, N=512)
        self.assertGreater(len(prices), 0)
        # Price at K=100 should be reasonable
        # Find closest strike to 100
        closest_k = min(prices.keys(), key=lambda k: abs(k - 100.0))
        self.assertGreater(prices[closest_k], 0)

    def test_batch_heston_price_jit(self):
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 110.0])
        maturities = np.array([1.0, 1.0])
        rates = np.array([0.05, 0.05])
        v0s = np.array([0.04, 0.04])
        kappas = np.array([2.0, 2.0])
        thetas = np.array([0.04, 0.04])
        sigmas = np.array([0.1, 0.1])
        rhos = np.array([-0.7, -0.7])
        is_calls = np.array([True, True])
        out = np.zeros(2)
        
        batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out)
        self.assertGreater(out[0], 0)
        self.assertGreater(out[0], out[1])

if __name__ == '__main__':
    unittest.main()
