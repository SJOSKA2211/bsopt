import pytest

from src.math_kernel.monte_carlo import BSParameters, MCConfig, MonteCarloEngine


class TestMonteCarlo:
    def setUp(self):
        self.params = BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.0,
        )
        self.config = MCConfig(n_paths=1000, seed=42)
        self.engine = MonteCarloEngine(self.config)

    def test_price_european(self):
        price, ci = self.engine.price_european(self.params, "call")
        self.assertGreater(price, 0)
        self.assertGreater(ci, 0)

    def test_calculate_greeks(self):
        greeks = self.engine.calculate_greeks(self.params, "call")
        self.assertGreater(greeks.delta, 0)
        self.assertLess(greeks.delta, 1.0)

    def test_price_american_lsm(self):
        price = self.engine.price_american_lsm(self.params, "call")
        self.assertGreater(price, 0)

    def test_sobol_method(self):
        config = MCConfig(n_paths=1024, method="sobol")
        engine = MonteCarloEngine(config)
        price, ci = engine.price_european(self.params, "call")
        self.assertGreater(price, 0)

