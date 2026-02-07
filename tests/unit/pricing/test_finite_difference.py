import pytest

from src.pricing.black_scholes import BSParameters
from src.pricing.finite_difference import CrankNicolsonSolver


def test_fd_price_european():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)

    price = solver.price(params, "call")
    assert price > 0

    # Compare with BS
    from src.pricing.black_scholes import BlackScholesEngine

    bs_price = BlackScholesEngine.price_options(params=params, option_type="call")
    assert pytest.approx(price, rel=1e-2) == bs_price


def test_fd_greeks():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    solver = CrankNicolsonSolver(n_spots=50, n_time=50)
    greeks = solver.calculate_greeks(params, "call")

    assert isinstance(greeks.delta, float)
    assert 0 < greeks.delta < 1


def test_fd_zero_maturity():
    params = BSParameters(
        spot=110.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05
    )
    solver = CrankNicolsonSolver()
    price = solver.price(params, "call")
    assert price == 10.0


def test_fd_diagnostics():
    solver = CrankNicolsonSolver()
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    solver._setup_grid(params)
    diag = solver.get_diagnostics()
    assert diag["scheme"] == "Crank-Nicolson"
    assert "stability" in diag
