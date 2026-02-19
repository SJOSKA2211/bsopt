import pytest

from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.models import BSParameters


@pytest.fixture
def sample_params():
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )


def test_cn_solver_price(sample_params):
    solver = CrankNicolsonSolver(n_spots=100, n_time=50)
    price = solver.price(sample_params, option_type="call")
    assert price > 0
    # Should be close to BS price (~10.45)
    assert 10.0 < price < 11.0


def test_cn_solver_greeks(sample_params):
    solver = CrankNicolsonSolver(n_spots=100, n_time=50)
    greeks = solver.calculate_greeks(sample_params, option_type="call")
    assert greeks.delta > 0
    assert greeks.gamma > 0
    assert greeks.vega > 0


def test_zero_maturity(sample_params):
    sample_params.maturity = 0.0
    solver = CrankNicolsonSolver()
    price = solver.price(sample_params, "call")
    assert price == 0.0  # ATM call

    sample_params.spot = 110.0
    price = solver.price(sample_params, "call")
    assert price == 10.0

    greeks = solver.calculate_greeks(sample_params, "call")
    assert greeks.delta == 1.0


def test_get_diagnostics(sample_params):
    solver = CrankNicolsonSolver()
    solver._setup_grid(sample_params)
    diag = solver.get_diagnostics()
    assert diag["scheme"] == "Crank-Nicolson"
    assert "stability" in diag


def test_clone():
    solver = CrankNicolsonSolver(n_spots=100)
    cloned = solver._clone(n_spots=200)
    assert cloned.n_spots == 200
    assert cloned.n_time == 500  # Default preserved
