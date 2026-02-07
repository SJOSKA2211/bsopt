import numpy as np
import pytest

from src.pricing.black_scholes import BSParameters
from src.pricing.lattice import (
    BinomialTreePricer,
    TrinomialTreePricer,
    validate_convergence,
)


def test_binomial_european():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=100, exercise_type="european")

    price = pricer.price(params, "call")
    assert price > 0
    # For European, should be close to Black-Scholes
    from src.pricing.black_scholes import BlackScholesEngine

    bs_price = BlackScholesEngine.price_options(params=params, option_type="call")
    assert pytest.approx(price, rel=1e-2) == bs_price


def test_binomial_american():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=100, exercise_type="american")

    price = pricer.price(params, "put")
    assert price > 0

    # American put should be more than European put
    eur_pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
    eur_price = eur_pricer.price(params, "put")
    assert price >= eur_price


def test_trinomial_european():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="european")

    price = pricer.price(params, "call")
    assert price > 0

    from src.pricing.black_scholes import BlackScholesEngine

    bs_price = float(
        np.atleast_1d(
            BlackScholesEngine.price_options(params=params, option_type="call")
        )[0]
    )
    assert pytest.approx(price, rel=1e-2) == bs_price


def test_lattice_greeks():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=50)
    greeks = pricer.calculate_greeks(params, "call")

    assert isinstance(greeks.delta, float)
    assert greeks.delta > 0


def test_validate_convergence():
    results = validate_convergence(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call", [10, 20])
    assert "binomial_errors" in results
    assert len(results["binomial_errors"]) == 2


def test_binomial_zero_maturity():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=10)
    assert pricer.price(params, "call") == 0.0

    params_itm = BSParameters(
        spot=110.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05
    )
    assert pricer.price(params_itm, "call") == 10.0


def test_trinomial_zero_maturity():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05
    )
    pricer = TrinomialTreePricer(n_steps=10)
    assert pricer.price(params, "call") == 0.0


def test_binomial_high_volatility():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=2.0, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=100)
    price = pricer.price(params, "call")
    assert price > 0
    assert price < 100.0


def test_build_tree():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    pricer = BinomialTreePricer(n_steps=2)
    tree = pricer.build_tree(params)
    assert tree.shape == (3, 3)
    assert tree[0, 0] == 100.0
