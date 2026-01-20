import pytest
import numpy as np
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.models import BSParameters

def test_fdm_solver_basic_pricing():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    
    price_call = solver.price(params, "call")
    assert price_call > 0
    
    price_put = solver.price(params, "put")
    assert price_put > 0

def test_fdm_solver_zero_maturity():
    params = BSParameters(100, 100, 0.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    
    assert solver.price(params, "call") == 0.0
    assert solver.price(params, "put") == 0.0
    
    # ITM
    params_itm = BSParameters(110, 100, 0.0, 0.2, 0.05)
    assert solver.price(params_itm, "call") == 10.0
    assert solver.price(params_itm, "put") == 0.0
    
    # ITM put
    params_itm_put = BSParameters(90, 100, 0.0, 0.2, 0.05)
    assert solver.price(params_itm_put, "put") == 10.0

def test_fdm_solver_solve_method():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    solver._setup_grid(params)
    solver.option_type = "call"
    price = solver.solve()
    assert price > 0
    
    # Zero maturity solve
    params_zero = BSParameters(100, 100, 0.0, 0.2, 0.05)
    solver._setup_grid(params_zero)
    assert solver.solve() == 0.0

def test_fdm_solver_greeks():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=50, n_time=50)
    greeks = solver.calculate_greeks(params, "call")
    assert greeks.delta > 0
    assert greeks.gamma > 0
    assert greeks.vega > 0
    assert greeks.theta <= 0
    assert greeks.rho > 0

def test_fdm_solver_greeks_zero_maturity():
    solver = CrankNicolsonSolver()
    
    # Call OTM
    params = BSParameters(90, 100, 0.0, 0.2, 0.05)
    greeks = solver.calculate_greeks(params, "call")
    assert greeks.delta == 0.0
    
    # Call ITM
    params_itm = BSParameters(110, 100, 0.0, 0.2, 0.05)
    greeks_itm = solver.calculate_greeks(params_itm, "call")
    assert greeks_itm.delta == 1.0
    
    # Put ITM
    params_put_itm = BSParameters(90, 100, 0.0, 0.2, 0.05)
    greeks_put = solver.calculate_greeks(params_put_itm, "put")
    assert greeks_put.delta == -1.0

def test_fdm_solver_iterative():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(use_iterative=True, n_spots=50, n_time=50)
    price = solver.price(params, "call")
    assert price > 0

def test_fdm_solver_diagnostics_and_clone():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    solver.price(params, "call")
    diag = solver.get_diagnostics()
    assert diag["scheme"] == "Crank-Nicolson"
    
    cloned = solver._clone(n_spots=300)
    assert cloned.n_spots == 300
    assert cloned.n_time == solver.n_time

def test_fdm_solver_greeks_edge_idx():
    # Spot near max
    solver = CrankNicolsonSolver(n_spots=10)
    params_high = BSParameters(299, 100, 1.0, 0.2, 0.05)
    greeks_high = solver.calculate_greeks(params_high, "call")
    assert greeks_high.delta >= 0

def test_fdm_solver_iterative_failure(mocker):
    # Patch cg in the module where it's used
    mocker.patch("src.pricing.finite_difference.cg", return_value=(np.zeros(199), 1))
    
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(use_iterative=True, n_spots=200, n_time=10)
    price = solver.price(params, "call")
    assert price >= 0

def test_fdm_solver_ilu_failure(mocker):
    mocker.patch("src.pricing.finite_difference.spilu", side_effect=RuntimeError("ILU failed"))
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(use_iterative=True, n_spots=50, n_time=50)
    price = solver.price(params, "call")
    assert price > 0

def test_fdm_solver_greeks_zero_spot():
    # Spot = 0.0 to hit idx == 0 logic
    params = BSParameters(0.0, 100, 1.0, 0.2, 0.05)
    solver = CrankNicolsonSolver(n_spots=10)
    greeks = solver.calculate_greeks(params, "call")
    assert greeks.delta >= 0.0

def test_fdm_solver_solve_zero_maturity_call():
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    solver._setup_grid(params)
    solver.option_type = "call"
    assert solver.solve() == 10.0

def test_fdm_solver_solve_zero_maturity_put():
    params = BSParameters(90, 100, 0.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    solver._setup_grid(params)
    solver.option_type = "put"
    assert solver.solve() == 10.0

def test_fdm_solver_get_greeks_small_maturity():
    params = BSParameters(100, 100, 0.5 / 365.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(params, "call")
    assert greeks.theta == 0.0

def test_fdm_solver_greeks_zero_maturity_put_otm():
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(params, "put")
    assert greeks.delta == 0.0