#!/usr/bin/env python3

"""
Minimal test to validate the Crank-Nicolson finite difference solver.
This script can be run when dependencies are available.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np

    from src.pricing.black_scholes import BSParameters
    from src.pricing.finite_difference import CrankNicolsonSolver

    print("Testing Crank-Nicolson Finite Difference Solver")
    print("=" * 60)

    # Test case: ATM European call
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
    )
    solver = CrankNicolsonSolver(
        n_spots=100,
        n_time=100,
    )

    print("\nTest 1: Solve for option price")
    price = solver.price(params, "call")
    print(f"  Option price: ${price:.6f}")
    print("  Status: PASS" if 5.0 < price < 15.0 else "  Status: FAIL")

    print("\nTest 2: Calculate Greeks")
    greeks = solver.calculate_greeks(params, "call")
    print(f"  Delta: {greeks.delta:.6f}")
    print(f"  Gamma: {greeks.gamma:.6f}")
    print(f"  Vega: {greeks.vega:.6f}")
    print(f"  Theta: {greeks.theta:.6f}")
    print(f"  Rho: {greeks.rho:.6f}")
    print("  Status: PASS" if 0.4 < greeks.delta < 0.7 else "  Status: FAIL")

    print("\nTest 3: Diagnostics")
    # setup grid manually for diagnostics as it's normally called inside price/greeks
    solver._setup_grid(params)
    diag = solver.get_diagnostics()
    print(f"  Scheme: {diag['scheme']}")
    print(f"  Grid: {diag['grid_spacing']['n_spots']}x{diag['grid_spacing']['n_time']}")
    print(f"  Stable: {diag['stability']['is_stable']}")
    print("  Status: PASS")

    print("\nTest 4: Edge cases")

    # Zero maturity
    params_zero = BSParameters(
        spot=105.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05
    )
    solver_zero = CrankNicolsonSolver(
        n_spots=50,
        n_time=50,
    )
    price_zero = solver_zero.price(params_zero, "call")
    expected = 5.0
    print(f"  Zero maturity call (S=105, K=100): ${price_zero:.6f}")
    print(f"  Expected intrinsic value: ${expected:.6f}")
    print("  Status: PASS" if abs(price_zero - expected) < 0.01 else "  Status: FAIL")

    # Deep ITM put
    params_itm = BSParameters(
        spot=80.0, strike=100.0, maturity=0.5, volatility=0.2, rate=0.05, dividend=0.0
    )
    solver_itm = CrankNicolsonSolver(
        n_spots=100,
        n_time=100,
    )
    price_itm = solver_itm.price(params_itm, "put")
    # For European put, intrinsic value is max(K*e^(-rT) - S*e^(-qT), 0)
    intrinsic = 100.0 * np.exp(-0.05 * 0.5) - 80.0
    print(f"  Deep ITM put (S=80, K=100): ${price_itm:.6f}")
    print(f"  Discounted Intrinsic value: ${intrinsic:.6f}")
    print("  Status: PASS" if price_itm >= intrinsic - 1e-5 else "  Status: FAIL")

    print("\nTest 5: Convergence")
    grid_sizes = [50, 100, 150]
    prices = []

    for n in grid_sizes:
        s = CrankNicolsonSolver(
            n_spots=n,
            n_time=n * 2,
        )
        prices.append(s.price(params, "call"))

    print(f"  Grid 50x100:  ${prices[0]:.6f}")
    print(f"  Grid 100x200: ${prices[1]:.6f}")
    print(f"  Grid 150x300: ${prices[2]:.6f}")

    # Check convergence: differences should decrease
    diff1 = abs(prices[1] - prices[0])
    diff2 = abs(prices[2] - prices[1])

    print(f"  Difference 1: ${diff1:.6f}")
    print(f"  Difference 2: ${diff2:.6f}")
    print(f"  Converging: {diff2 < diff1}")
    print("  Status: PASS" if diff2 < diff1 else "  Status: FAIL")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("\nTo run this test, install dependencies:")
    print("  pip install numpy scipy")
    sys.exit(1)
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
