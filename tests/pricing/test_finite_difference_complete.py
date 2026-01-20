import pytest
import numpy as np
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.models import BSParameters, OptionGreeks

# Constants
SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
VOLATILITY = 0.2
RATE = 0.05
DIVIDEND = 0.0

@pytest.fixture
def bs_params():
    return BSParameters(SPOT, STRIKE, MATURITY, VOLATILITY, RATE, DIVIDEND)

def test_fd_solver_init():
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    assert solver.n_spots == 100
    assert solver.n_time == 100

def test_price_call(bs_params):
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    price = solver.price(bs_params, "call")
    
    # BS Price ~ 10.45
    assert 10.0 < price < 11.0

def test_price_put(bs_params):
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    price = solver.price(bs_params, "put")
    
    # BS Price ~ 5.57
    assert 5.0 < price < 6.0

def test_price_zero_maturity(bs_params):
    bs_params.maturity = 0.0
    solver = CrankNicolsonSolver()
    
    price = solver.price(bs_params, "call")
    assert price == 0.0 # ATM
    
    bs_params.spot = 110.0
    price_itm = solver.price(bs_params, "call")
    assert price_itm == 10.0

def test_greeks_calculation(bs_params):
    # Increase n_time to capture small time decay (theta)
    solver = CrankNicolsonSolver(n_spots=100, n_time=100)
    greeks = solver.calculate_greeks(bs_params, "call")
    
    assert isinstance(greeks, OptionGreeks)
    # Delta for ATM call ~ 0.5 - 0.6
    assert 0.5 < greeks.delta < 0.7
    assert greeks.gamma > 0
    assert greeks.vega > 0
    # Theta should be negative (time decay) or close to zero
    assert greeks.theta <= 0
    assert greeks.rho > 0

def test_iterative_solver(bs_params):
    solver = CrankNicolsonSolver(n_spots=50, n_time=50, use_iterative=True)
    price = solver.price(bs_params, "call")
    assert 10.0 < price < 11.0

def test_diagnostics(bs_params):
    solver = CrankNicolsonSolver()
    solver._setup_grid(bs_params)
    diag = solver.get_diagnostics()
    
    assert diag["scheme"] == "Crank-Nicolson"
    assert "stability" in diag
    assert diag["stability"]["is_stable"] is True

def test_clone():
    solver = CrankNicolsonSolver(n_spots=50)
    cloned = solver._clone(n_spots=100)
    assert cloned.n_spots == 100
    assert solver.n_spots == 50

def test_zero_volatility_check(bs_params):
    # Should probably handle it gracefully or raise error
    # The code divides by dS^2 which is fine, but check max_explicit_dt
    bs_params.volatility = 0.0001 # Close to zero
    solver = CrankNicolsonSolver()
    price = solver.price(bs_params, "call")
    # Should converge to intrinsic discounted
    assert price > 0

def test_greeks_zero_maturity(bs_params):
    bs_params.maturity = 0.0
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(bs_params, "call")
    # Delta should be 0 (ATM) or 1 (ITM)
    assert greeks.delta == 0.0 # ATM spot=strike
    
    bs_params.spot = 110.0
    greeks_itm = solver.calculate_greeks(bs_params, "call")
    assert greeks_itm.delta == 1.0
    
    # Put case
    greeks_put = solver.calculate_greeks(bs_params, "put")
    assert greeks_put.delta == 0.0 # ITM call -> OTM Put -> 0? Wait S=110, K=100. Put OTM. Delta 0.
    
    bs_params.spot = 90.0
    greeks_put_itm = solver.calculate_greeks(bs_params, "put")
    assert greeks_put_itm.delta == -1.0

def test_solve_method(bs_params):
    solver = CrankNicolsonSolver()
    solver._setup_grid(bs_params)
    solver.option_type = "call"
    price = solver.solve()
    assert price > 0

    solver.maturity = 0.0
    price_zero = solver.solve()
    assert price_zero == 0.0

def test_short_maturity_theta(bs_params):
    # Maturity < 1 day
    bs_params.maturity = 0.0001
    solver = CrankNicolsonSolver()
    greeks = solver.calculate_greeks(bs_params, "call")
    assert greeks.theta == 0.0

from unittest.mock import patch, MagicMock

def test_solver_error_handling(bs_params):
    solver = CrankNicolsonSolver(use_iterative=True)
    
    # Mock spilu to raise RuntimeError
    with patch("src.pricing.finite_difference.spilu", side_effect=RuntimeError("Fail")):
        price = solver.price(bs_params, "call")
        assert price > 0 # Should fallback to direct solver

def test_cg_convergence_warning(bs_params):
    solver = CrankNicolsonSolver(use_iterative=True)
    
    # Mock cg to return info=1 (failure)
    # cg returns (x, info)
    with patch("src.pricing.finite_difference.cg", return_value=(np.zeros(199), 1)):
        # We need to ensure it runs without crashing, but logs warning
        # Since we mock cg, the result will be garbage (zeros), so price might be wrong but flow completes.
        solver.price(bs_params, "call")
        # Just checking it doesn't raise exception
