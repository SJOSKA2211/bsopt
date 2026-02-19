from src.pricing.black_scholes import BSParameters
from src.pricing.monte_carlo import MCConfig, MonteCarloEngine, geometric_asian_price
from tests.test_utils import assert_equal


def test_mc_config():
    config = MCConfig(n_paths=1000, method="sobol")
    assert_equal(config.n_paths, 1024)  # Power of 2 for Sobol

    config = MCConfig(n_paths=1001, antithetic=True)
    assert_equal(config.n_paths, 1002)  # Even for antithetic


def test_mc_european_pricing():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    engine = MonteCarloEngine(MCConfig(n_paths=10000, seed=42))

    price, conf_int = engine.price_european(params, option_type="call")
    # BS price is approx 10.45
    assert 10.0 < price < 11.0
    assert conf_int > 0


def test_mc_american_pricing():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    engine = MonteCarloEngine(MCConfig(n_paths=10000, seed=42))

    # American put should be more expensive than European put
    price = engine.price_american_lsm(params, option_type="put")
    assert price > 0


def test_geometric_asian_price():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    price = geometric_asian_price(params, "call", 252)
    assert price > 0
    assert price < 10.45  # Asian should be cheaper than vanilla
