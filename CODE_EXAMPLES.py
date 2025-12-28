#!/usr/bin/env python3
"""
Code Examples for Crank-Nicolson Finite Difference Solver

This file contains practical examples demonstrating various use cases
of the CrankNicolsonSolver for option pricing and Greeks calculation.

Note: These examples assume numpy and scipy are installed.
Install with: pip install numpy scipy
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def example_1_basic_call_option():
    """
    Example 1: Price a European call option.

    Parameters:
    - ATM option (S=K=100)
    - 1 year to maturity
    - 20% volatility
    - 5% risk-free rate
    - 2% dividend yield
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic European Call Option")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    solver = CrankNicolsonSolver(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.02,
        option_type='call',
        n_spots=200,
        n_time=500
    )

    price = solver.solve()

    print(f"\nOption Parameters:")
    print(f"  Spot (S):      ${solver.spot:.2f}")
    print(f"  Strike (K):    ${solver.strike:.2f}")
    print(f"  Maturity (T):  {solver.maturity:.2f} years")
    print(f"  Volatility (σ): {solver.volatility:.1%}")
    print(f"  Rate (r):      {solver.rate:.1%}")
    print(f"  Dividend (q):  {solver.dividend:.1%}")

    print(f"\nGrid Configuration:")
    print(f"  Spatial points: {solver.n_spots}")
    print(f"  Time steps:     {solver.n_time}")
    print(f"  Grid spacing:   dS={solver.dS:.4f}, dt={solver.dt:.6f}")

    print(f"\nResult:")
    print(f"  Call Option Price: ${price:.4f}")
    print()


def example_2_put_option_with_greeks():
    """
    Example 2: Price a European put option and calculate Greeks.

    Parameters:
    - OTM put (S=110, K=100)
    - 6 months to maturity
    - 25% volatility
    - 3% risk-free rate
    - No dividends
    """
    print("=" * 70)
    print("EXAMPLE 2: European Put Option with Greeks")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    solver = CrankNicolsonSolver(
        spot=110.0,
        strike=100.0,
        maturity=0.5,
        volatility=0.25,
        rate=0.03,
        dividend=0.0,
        option_type='put',
        n_spots=150,
        n_time=300
    )

    price = solver.solve()
    greeks = solver.get_greeks()

    print(f"\nOption Parameters:")
    print(f"  Type: PUT")
    print(f"  Spot (S):      ${solver.spot:.2f}")
    print(f"  Strike (K):    ${solver.strike:.2f}")
    print(f"  Maturity (T):  {solver.maturity:.2f} years")
    print(f"  Volatility (σ): {solver.volatility:.1%}")
    print(f"  Rate (r):      {solver.rate:.1%}")

    print(f"\nPricing Results:")
    print(f"  Put Option Price: ${price:.4f}")
    print(f"  Intrinsic Value:  ${max(solver.strike - solver.spot, 0):.4f}")
    print(f"  Time Value:       ${price - max(solver.strike - solver.spot, 0):.4f}")

    print(f"\nGreeks:")
    print(f"  Delta (∂V/∂S):    {greeks.delta:>8.4f}")
    print(f"  Gamma (∂²V/∂S²):  {greeks.gamma:>8.6f}")
    print(f"  Vega (∂V/∂σ):     {greeks.vega:>8.4f}")
    print(f"  Theta (∂V/∂t):    {greeks.theta:>8.4f} per day")
    print(f"  Rho (∂V/∂r):      {greeks.rho:>8.4f} per 1%")
    print()


def example_3_itm_call_with_dividends():
    """
    Example 3: Deep ITM call on dividend-paying stock.

    Parameters:
    - ITM call (S=120, K=100)
    - 3 months to maturity
    - 30% volatility
    - 4% risk-free rate
    - 3% dividend yield
    """
    print("=" * 70)
    print("EXAMPLE 3: Deep ITM Call with Dividends")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    solver = CrankNicolsonSolver(
        spot=120.0,
        strike=100.0,
        maturity=0.25,
        volatility=0.30,
        rate=0.04,
        dividend=0.03,
        option_type='call',
        n_spots=200,
        n_time=400
    )

    price = solver.solve()
    intrinsic = max(solver.spot - solver.strike, 0)

    print(f"\nOption Parameters:")
    print(f"  Type: CALL (Deep ITM)")
    print(f"  Spot (S):      ${solver.spot:.2f}")
    print(f"  Strike (K):    ${solver.strike:.2f}")
    print(f"  Moneyness:     {(solver.spot / solver.strike - 1) * 100:.1f}% ITM")
    print(f"  Maturity (T):  {solver.maturity * 12:.1f} months")
    print(f"  Volatility (σ): {solver.volatility:.1%}")
    print(f"  Rate (r):      {solver.rate:.1%}")
    print(f"  Dividend (q):  {solver.dividend:.1%}")

    print(f"\nPricing Results:")
    print(f"  Call Option Price: ${price:.4f}")
    print(f"  Intrinsic Value:   ${intrinsic:.4f}")
    print(f"  Time Value:        ${price - intrinsic:.4f}")
    print(f"  Parity Check:      S - K = ${solver.spot - solver.strike:.4f}")
    print()


def example_4_convergence_study():
    """
    Example 4: Convergence study with different grid sizes.

    Demonstrates how the solution converges as the grid is refined.
    """
    print("=" * 70)
    print("EXAMPLE 4: Grid Convergence Study")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver
    import time

    # Fixed parameters
    params = {
        'spot': 100.0,
        'strike': 100.0,
        'maturity': 1.0,
        'volatility': 0.2,
        'rate': 0.05,
        'dividend': 0.02,
        'option_type': 'call'
    }

    # Different grid configurations
    grids = [
        (50, 100),
        (100, 200),
        (150, 300),
        (200, 500),
        (300, 600)
    ]

    print(f"\nFixed Parameters:")
    print(f"  Spot: ${params['spot']:.2f}")
    print(f"  Strike: ${params['strike']:.2f}")
    print(f"  Maturity: {params['maturity']:.2f} years")
    print(f"  Volatility: {params['volatility']:.1%}")

    print(f"\nConvergence Study:")
    print(f"{'Grid Size':<15} {'Price':<15} {'Change':<15} {'Time (ms)':<15}")
    print("-" * 60)

    prev_price = None

    for n_spots, n_time in grids:
        start = time.time()

        solver = CrankNicolsonSolver(
            **params,
            n_spots=n_spots,
            n_time=n_time
        )

        price = solver.solve()
        elapsed = (time.time() - start) * 1000

        if prev_price is not None:
            change = abs(price - prev_price)
            print(f"{n_spots}x{n_time:<10} ${price:<14.6f} ${change:<14.6f} {elapsed:<14.2f}")
        else:
            print(f"{n_spots}x{n_time:<10} ${price:<14.6f} {'N/A':<14} {elapsed:<14.2f}")

        prev_price = price

    print("\nObservation: Price should converge as grid is refined.")
    print()


def example_5_put_call_parity():
    """
    Example 5: Verify put-call parity.

    Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
    """
    print("=" * 70)
    print("EXAMPLE 5: Put-Call Parity Verification")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver
    import math

    # Common parameters
    params = {
        'spot': 100.0,
        'strike': 100.0,
        'maturity': 1.0,
        'volatility': 0.2,
        'rate': 0.05,
        'dividend': 0.02,
        'n_spots': 200,
        'n_time': 500
    }

    # Solve for call
    call_solver = CrankNicolsonSolver(**params, option_type='call')
    call_price = call_solver.solve()

    # Solve for put
    put_solver = CrankNicolsonSolver(**params, option_type='put')
    put_price = put_solver.solve()

    # Calculate theoretical parity
    S = params['spot']
    K = params['strike']
    T = params['maturity']
    r = params['rate']
    q = params['dividend']

    theoretical = S * math.exp(-q * T) - K * math.exp(-r * T)
    actual = call_price - put_price
    error = abs(actual - theoretical)

    print(f"\nOption Parameters:")
    print(f"  Spot (S):      ${S:.2f}")
    print(f"  Strike (K):    ${K:.2f}")
    print(f"  Maturity (T):  {T:.2f} years")
    print(f"  Rate (r):      {r:.1%}")
    print(f"  Dividend (q):  {q:.1%}")

    print(f"\nPricing Results:")
    print(f"  Call Price (C): ${call_price:.6f}")
    print(f"  Put Price (P):  ${put_price:.6f}")

    print(f"\nPut-Call Parity Check:")
    print(f"  C - P (actual):      ${actual:.6f}")
    print(f"  S*e^(-qT) - K*e^(-rT) (theoretical): ${theoretical:.6f}")
    print(f"  Absolute Error:      ${error:.6f}")
    print(f"  Relative Error:      {error / theoretical * 100:.3f}%")

    if error / theoretical < 0.01:
        print("\n  ✓ Put-Call Parity VERIFIED (error < 1%)")
    else:
        print("\n  ✗ Put-Call Parity error exceeds 1% (refine grid)")
    print()


def example_6_zero_maturity():
    """
    Example 6: Zero maturity (at expiration).

    At maturity, option value should equal intrinsic value.
    """
    print("=" * 70)
    print("EXAMPLE 6: Zero Maturity (At Expiration)")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    test_cases = [
        (105, 100, 'call', 5.0),    # ITM call
        (95, 100, 'call', 0.0),     # OTM call
        (105, 100, 'put', 0.0),     # OTM put
        (95, 100, 'put', 5.0),      # ITM put
    ]

    print(f"\n{'Spot':<10} {'Strike':<10} {'Type':<8} {'Price':<12} {'Expected':<12} {'Status':<10}")
    print("-" * 62)

    for spot, strike, opt_type, expected in test_cases:
        solver = CrankNicolsonSolver(
            spot=spot,
            strike=strike,
            maturity=0.0,  # Zero maturity
            volatility=0.2,
            rate=0.05,
            dividend=0.0,
            option_type=opt_type,
            n_spots=100,
            n_time=100
        )

        price = solver.solve()
        error = abs(price - expected)
        status = "PASS" if error < 0.01 else "FAIL"

        print(f"${spot:<9.2f} ${strike:<9.2f} {opt_type.upper():<8} ${price:<11.6f} ${expected:<11.2f} {status:<10}")

    print("\nAt maturity (T=0), option value equals intrinsic value.")
    print()


def example_7_sensitivity_analysis():
    """
    Example 7: Sensitivity to volatility and interest rate.

    Demonstrates how option price changes with different parameters.
    """
    print("=" * 70)
    print("EXAMPLE 7: Parameter Sensitivity Analysis")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    base_params = {
        'spot': 100.0,
        'strike': 100.0,
        'maturity': 1.0,
        'volatility': 0.2,
        'rate': 0.05,
        'dividend': 0.0,
        'option_type': 'call',
        'n_spots': 150,
        'n_time': 300
    }

    # Volatility sensitivity
    print("\nVolatility Sensitivity:")
    print(f"{'Volatility':<15} {'Call Price':<15} {'Change %':<15}")
    print("-" * 45)

    volatilities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    base_price = None

    for vol in volatilities:
        params = base_params.copy()
        params['volatility'] = vol

        solver = CrankNicolsonSolver(**params)
        price = solver.solve()

        if base_price is None:
            base_price = price
            change_pct = 0.0
        else:
            change_pct = (price / base_price - 1) * 100

        print(f"{vol:<14.1%} ${price:<14.4f} {change_pct:>+6.2f}%")

    # Interest rate sensitivity
    print("\nInterest Rate Sensitivity:")
    print(f"{'Rate':<15} {'Call Price':<15} {'Change %':<15}")
    print("-" * 45)

    rates = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    base_price = None

    for rate in rates:
        params = base_params.copy()
        params['rate'] = rate

        solver = CrankNicolsonSolver(**params)
        price = solver.solve()

        if base_price is None:
            base_price = price
            change_pct = 0.0
        else:
            change_pct = (price / base_price - 1) * 100

        print(f"{rate:<14.1%} ${price:<14.4f} {change_pct:>+6.2f}%")

    print()


def example_8_diagnostics():
    """
    Example 8: Solver diagnostics and configuration details.
    """
    print("=" * 70)
    print("EXAMPLE 8: Solver Diagnostics")
    print("=" * 70)

    from pricing.finite_difference import CrankNicolsonSolver

    solver = CrankNicolsonSolver(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.02,
        option_type='call',
        n_spots=200,
        n_time=500
    )

    diag = solver.get_diagnostics()

    print(f"\nScheme Information:")
    print(f"  Method: {diag['scheme']}")

    print(f"\nGrid Configuration:")
    print(f"  Spatial points (M+1): {diag['grid_spacing']['n_spots']}")
    print(f"  Time steps (N):       {diag['grid_spacing']['n_time']}")
    print(f"  Spatial step (dS):    {diag['grid_spacing']['dS']:.6f}")
    print(f"  Temporal step (dt):   {diag['grid_spacing']['dt']:.6f}")

    print(f"\nDomain Coverage:")
    print(f"  S ∈ [{diag['domain']['S_min']:.2f}, {diag['domain']['S_max']:.2f}]")
    print(f"  T = {diag['domain']['T']:.2f} years")

    print(f"\nStability Analysis:")
    print(f"  Mesh ratio (dt/dS²):   {diag['stability']['mesh_ratio']:.8f}")
    print(f"  Max explicit dt:       {diag['stability']['max_explicit_dt']:.8f}")
    print(f"  Is stable:             {diag['stability']['is_stable']}")
    print(f"  Note: {diag['stability']['note']}")

    print(f"\nAccuracy Characteristics:")
    print(f"  Spatial order:  {diag['accuracy']['spatial_order']}")
    print(f"  Temporal order: {diag['accuracy']['temporal_order']}")
    print(f"  Note: {diag['accuracy']['note']}")

    print(f"\nPerformance:")
    print(f"  Operations per step: {diag['performance']['operations_per_step']}")
    print(f"  Total operations:    {diag['performance']['total_operations']}")
    print(f"  Sparse matrix:       {diag['performance']['sparse_matrix']}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 76 + "╗")
    print("║" + " " * 15 + "CRANK-NICOLSON SOLVER - CODE EXAMPLES" + " " * 24 + "║")
    print("╚" + "=" * 76 + "╝")
    print("\n")

    try:
        # Check if dependencies are available
        import numpy
        import scipy

        # Run examples
        example_1_basic_call_option()
        input("Press Enter to continue to Example 2...")

        example_2_put_option_with_greeks()
        input("Press Enter to continue to Example 3...")

        example_3_itm_call_with_dividends()
        input("Press Enter to continue to Example 4...")

        example_4_convergence_study()
        input("Press Enter to continue to Example 5...")

        example_5_put_call_parity()
        input("Press Enter to continue to Example 6...")

        example_6_zero_maturity()
        input("Press Enter to continue to Example 7...")

        example_7_sensitivity_analysis()
        input("Press Enter to continue to Example 8...")

        example_8_diagnostics()

        print("\n")
        print("╔" + "=" * 76 + "╗")
        print("║" + " " * 25 + "ALL EXAMPLES COMPLETED" + " " * 29 + "║")
        print("╚" + "=" * 76 + "╝")
        print("\n")

    except ImportError as e:
        print(f"\nError: Missing dependencies - {e}")
        print("\nTo run these examples, install the required packages:")
        print("  pip install numpy scipy")
        print()


if __name__ == "__main__":
    main()
