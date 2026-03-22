import unittest

from src.math_kernel.lattice import BinomialTreePricer, BSParameters, TrinomialTreePricer


class TestLattice(unittest.TestCase):
    def setUp(self):
        self.params = BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.01,
        )

    def test_binomial_pricer(self):
        pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
        price = pricer.price(self.params, option_type="call")
        self.assertGreater(price, 0)

        # American should be >= European
        am_pricer = BinomialTreePricer(n_steps=100, exercise_type="american")
        am_price = am_pricer.price(self.params, option_type="put")
        eu_price = pricer.price(self.params, option_type="put")
        self.assertGreaterEqual(am_price, eu_price)

    def test_trinomial_pricer(self):
        pricer = TrinomialTreePricer(n_steps=50, exercise_type="european")
        price = pricer.price(self.params, option_type="call")
        self.assertGreater(price, 0)

    def test_calculate_greeks(self):
        pricer = BinomialTreePricer(n_steps=50)
        greeks = pricer.calculate_greeks(self.params, option_type="call")
        self.assertGreater(greeks.delta, 0)

    def test_build_tree(self):
        pricer = BinomialTreePricer(n_steps=10)
        tree = pricer.build_tree(self.params)
        self.assertEqual(tree.shape, (11, 11))


if __name__ == "__main__":
    unittest.main()
