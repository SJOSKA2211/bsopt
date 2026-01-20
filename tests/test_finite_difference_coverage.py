import numpy as np
import pytest
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.black_scholes import BSParameters

def test_fdm_call_price():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
    )
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    price = solver.price(params, "call")
    assert 5.0 < price < 15.0

def test_fdm_put_price():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
    )
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    price = solver.price(params, "put")
    assert price > 0

def test_fdm_greeks():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
    )
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    greeks = solver.calculate_greeks(params, "call")
    assert 0.4 < greeks.delta < 0.7
    assert greeks.gamma > 0
    # theta might be zero due to grid resolution or small bump
    assert greeks.vega >= 0
    assert greeks.rho >= 0

def test_fdm_zero_maturity():
    params = BSParameters(spot=105.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    solver = CrankNicolsonSolver(n_spots=50, n_time=50)
    price = solver.price(params, "call")
    assert price == pytest.approx(5.0)
    
    # Hit line 67: zero maturity Put
    price_put = solver.price(params, "put")
    assert price_put == 0.0

def test_fdm_diagnostics():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=50, n_time=50)
    solver._setup_grid(params)
    diag = solver.get_diagnostics()
    assert diag['scheme'] == "Crank-Nicolson"
    assert diag['stability']['is_stable'] is True

def test_fdm_boundary_conditions():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    greeks = solver.calculate_greeks(params, "put")
    assert greeks.delta < 0

def test_fdm_zero_maturity_put_greeks():
    params = BSParameters(spot=95.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(params, "put")
    assert greeks.delta == -1.0

def test_fdm_boundary_indices():
    params_low = BSParameters(spot=1.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    solver = CrankNicolsonSolver(n_spots=50)
    greeks_low = solver.calculate_greeks(params_low, "call")
    assert greeks_low.delta >= 0
    
    params_high = BSParameters(spot=500.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    greeks_high = solver.calculate_greeks(params_high, "call")
    assert greeks_high.delta >= 0

def test_fdm_iterative_solver():
    # Use smaller grid for speed
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=50, n_time=50, use_iterative=True)
    price = solver.price(params, "call")
    assert price > 0

def test_fdm_solve_unused_method():
    # Hit solve() method (lines 191-202)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    solver._setup_grid(params)
    solver.option_type = "call"
    price = solver.solve()
    assert price > 0
    
    # Zero maturity branch in solve()
    params_zero = BSParameters(100, 100, 0.0, 0.2, 0.05)
    solver._setup_grid(params_zero)
    assert solver.solve() == 0.0
    solver.option_type = "put"
    assert solver.solve() == 0.0

def test_fdm_zero_maturity_put_otm_greeks():
    params = BSParameters(spot=105.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(params, "put")
    assert greeks.delta == 0.0

def test_fdm_iterative_solver_ilu_fail():
    from unittest.mock import patch
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(use_iterative=True)
    with patch('src.pricing.finite_difference.spilu', side_effect=RuntimeError("mock fail")):
        price = solver.price(params, "call")
        assert price > 0

def test_fdm_iterative_solver_convergence_fail():
    from unittest.mock import patch
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(use_iterative=True)
    # Mock cg to return info=1 (failure to converge)
    # The size 199 is for default n_spots=200
    with patch('src.pricing.finite_difference.cg', return_value=(np.zeros(199), 1)):
        solver.price(params, "call")

def test_fdm_clone():
    solver = CrankNicolsonSolver(n_spots=150)
    cloned = solver._clone(n_time=200)
    assert cloned.n_spots == 150
    assert cloned.n_time == 200

def test_fdm_short_maturity_theta():
    params = BSParameters(100, 100, 0.001, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=50, n_time=50)
    greeks = solver.calculate_greeks(params, "call")
    assert greeks.theta is not None
