import unittest

from src.quant.pricing.models import (
    BSParameters,
    HestonParams,
    OptionGreeks,
    global_model_pool,
)


class TestModels(unittest.TestCase):
    def test_bs_parameters_validation(self):
        # Valid
        p = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
        self.assertEqual(p.spot, 100)

        # Invalid spot
        with self.assertRaises(ValueError):
            BSParameters(spot=-1, strike=100, maturity=1, volatility=0.2, rate=0.05)

    def test_option_greeks(self):
        g = OptionGreeks(delta=0.5, gamma=0.02, theta=-0.01, vega=0.1, rho=0.05)
        self.assertEqual(g["delta"], 0.5)
        self.assertIn("gamma", g)

    def test_heston_params_validation(self):
        # Valid (Feller condition: 2 * 2 * 0.04 > 0.1^2 => 0.16 > 0.01)
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)
        self.assertEqual(p.v0, 0.04)

        # Feller violation (2 * 1 * 0.04 < 0.5^2 => 0.08 < 0.25)
        with self.assertRaises(ValueError):
            HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7)

    def test_model_pool(self):
        p1 = global_model_pool.get_bs_params(
            spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05
        )
        global_model_pool.release_bs_params(p1)
        p2 = global_model_pool.get_bs_params(
            spot=110, strike=100, maturity=1, volatility=0.2, rate=0.05
        )
        self.assertIs(p1, p2)  # Should be same object from pool
        self.assertEqual(p2.spot, 110)


if __name__ == "__main__":
    unittest.main()
