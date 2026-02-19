import pytest

from src.pricing.lattice import (
    BinomialTreePricer,
    TrinomialTreePricer,
    validate_convergence,
)
from src.pricing.models import BSParameters


@pytest.fixture
def sample_params():
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )


def test_binomial_european(sample_params):
    pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
    price = pricer.price(sample_params, option_type="call")
    assert price > 0

    greeks = pricer.calculate_greeks(sample_params, option_type="call")
    assert greeks.delta > 0
    assert greeks.gamma > 0


def test_binomial_american(sample_params):
    pricer = BinomialTreePricer(n_steps=100, exercise_type="american")
    price = pricer.price(sample_params, option_type="call")
    assert price > 0


def test_trinomial_european(sample_params):
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="european")
    price = pricer.price(sample_params, option_type="call")
    assert price > 0

    greeks = pricer.calculate_greeks(sample_params, option_type="call")
    assert greeks.delta > 0


def test_trinomial_american(sample_params):
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="american")
    price = pricer.price(sample_params, option_type="call")
    assert price > 0


def test_validate_convergence():
    res = validate_convergence(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.0,
        option_type="call",
        step_sizes=[10, 50],
    )
    assert "binomial_errors" in res
    assert len(res["binomial_errors"]) == 2


def test_build_tree(sample_params):
    pricer = BinomialTreePricer(n_steps=5)
    tree = pricer.build_tree(sample_params)
    assert tree.shape == (6, 6)
    assert tree[0, 0] == 100.0


def test_zero_maturity(sample_params):
    sample_params.maturity = 0.0
    pricer = BinomialTreePricer()
    assert pricer.price(sample_params, "call") == 0.0

    sample_params.spot = 110.0
    assert pricer.price(sample_params, "call") == 10.0
