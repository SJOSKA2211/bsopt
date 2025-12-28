"""
Comprehensive Examples: Lattice-Based Option Pricing

This script demonstrates the full capabilities of the binomial and trinomial
tree option pricing models, including:

1. Basic option pricing (American and European)
2. Greeks calculation and interpretation
3. Convergence analysis to Black-Scholes
4. Early exercise boundary detection
5. Performance benchmarking
6. Comparison between binomial and trinomial methods

Run this file to see the models in action:
    python examples/lattice_examples.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pricing.lattice import (
    BinomialTreePricer,
    TrinomialTreePricer,
    validate_convergence
)
from src.pricing.black_scholes import BSParameters, BlackScholesEngine


def example_1_basic_pricing():
    """Example 1: Basic option pricing with binomial tree."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Option Pricing - Binomial Tree")
    print("=" * 80)
    print()

    # Market parameters
    S0 = 100.0  # Current stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (1 year)
    sigma = 0.25  # Volatility (25%)
    r = 0.05    # Risk-free rate (5%)
    q = 0.02    # Dividend yield (2%)
    n_steps = 200

    print(f"Market Parameters:")
    print(f"  Spot Price (S0):      ${S0:.2f}")
    print(f"  Strike Price (K):     ${K:.2f}")
    print(f"  Time to Maturity (T): {T:.1f} year")
    print(f"  Volatility (σ):       {sigma:.1%}")
    print(f"  Risk-Free Rate (r):   {r:.1%}")
    print(f"  Dividend Yield (q):   {q:.1%}")
    print(f"  Tree Steps (N):       {n_steps}")
    print()

    # Price European call
    print("European Call Option:")
    print("-" * 40)
    euro_call = BinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='call', exercise_type='european',
        n_steps=n_steps
    )
    euro_call_price = euro_call.price()
    euro_call_greeks = euro_call.get_greeks()

    print(f"  Price:  ${euro_call_price:.4f}")
    print(f"  Delta:  {euro_call_greeks.delta:>8.4f}")
    print(f"  Gamma:  {euro_call_greeks.gamma:>8.4f}")
    print(f"  Vega:   {euro_call_greeks.vega:>8.4f}")
    print(f"  Theta:  {euro_call_greeks.theta:>8.4f}")
    print(f"  Rho:    {euro_call_greeks.rho:>8.4f}")
    print()

    # Price European put
    print("European Put Option:")
    print("-" * 40)
    euro_put = BinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='put', exercise_type='european',
        n_steps=n_steps
    )
    euro_put_price = euro_put.price()
    euro_put_greeks = euro_put.get_greeks()

    print(f"  Price:  ${euro_put_price:.4f}")
    print(f"  Delta:  {euro_put_greeks.delta:>8.4f}")
    print(f"  Gamma:  {euro_put_greeks.gamma:>8.4f}")
    print(f"  Vega:   {euro_put_greeks.vega:>8.4f}")
    print(f"  Theta:  {euro_put_greeks.theta:>8.4f}")
    print(f"  Rho:    {euro_put_greeks.rho:>8.4f}")
    print()

    # Verify put-call parity
    forward = S0 * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    parity_lhs = euro_call_price - euro_put_price
    parity_rhs = forward - discounted_strike
    parity_error = abs(parity_lhs - parity_rhs)

    print("Put-Call Parity Verification:")
    print("-" * 40)
    print(f"  C - P = ${parity_lhs:.6f}")
    print(f"  S·e^(-qT) - K·e^(-rT) = ${parity_rhs:.6f}")
    print(f"  Difference: ${parity_error:.2e}")
    print(f"  Parity Holds: {parity_error < 1e-4}")
    print()


def example_2_american_options():
    """Example 2: American option pricing with early exercise."""
    print("=" * 80)
    print("EXAMPLE 2: American Options with Early Exercise Premium")
    print("=" * 80)
    print()

    # Parameters favorable for early exercise (high dividend, ITM put)
    S0 = 100.0
    K = 110.0  # Out-of-the-money for call, in-the-money for put
    T = 1.0
    sigma = 0.20
    r = 0.05
    q = 0.08  # High dividend yield
    n_steps = 300

    print(f"Parameters (High Dividend Yield):")
    print(f"  S0 = ${S0:.2f}, K = ${K:.2f}, T = {T:.1f} year")
    print(f"  σ = {sigma:.1%}, r = {r:.1%}, q = {q:.1%}")
    print(f"  N = {n_steps} steps")
    print()

    # American put
    print("American Put Option:")
    print("-" * 40)
    amer_put = BinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='put', exercise_type='american',
        n_steps=n_steps
    )
    amer_put_price = amer_put.price()

    # European put for comparison
    euro_put = BinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='put', exercise_type='european',
        n_steps=n_steps
    )
    euro_put_price = euro_put.price()

    early_premium = amer_put_price - euro_put_price

    print(f"  American Put Price: ${amer_put_price:.4f}")
    print(f"  European Put Price: ${euro_put_price:.4f}")
    print(f"  Early Exercise Premium: ${early_premium:.4f}")
    print(f"  Premium as % of European: {early_premium/euro_put_price:.2%}")
    print()

    # Get early exercise boundary
    boundary = amer_put.get_early_exercise_boundary()
    if boundary is not None:
        print("Early Exercise Boundary (selected time steps):")
        print("-" * 40)
        print(f"  {'Time':<10} {'Spot Boundary':<15} {'% of Strike':<15}")
        print("-" * 40)

        # Show boundary at select time points
        time_points = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]
        for i in time_points:
            t = i / n_steps * T
            if boundary[i] > 0:
                print(f"  {t:<10.3f} ${boundary[i]:<14.2f} {boundary[i]/K:<14.2%}")

    print()


def example_3_convergence_study():
    """Example 3: Convergence to Black-Scholes as steps increase."""
    print("=" * 80)
    print("EXAMPLE 3: Convergence Study - European Options")
    print("=" * 80)
    print()

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    sigma = 0.20
    r = 0.05
    q = 0.02

    print(f"Parameters: S0=${S0}, K=${K}, T={T}yr, σ={sigma:.0%}, r={r:.0%}, q={q:.0%}")
    print()

    # Black-Scholes reference
    bs_params = BSParameters(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q
    )
    bs_call = BlackScholesEngine.price_call(bs_params)
    bs_put = BlackScholesEngine.price_put(bs_params)

    print(f"Black-Scholes Prices (Analytical):")
    print(f"  Call: ${bs_call:.6f}")
    print(f"  Put:  ${bs_put:.6f}")
    print()

    # Convergence study
    step_sizes = [25, 50, 100, 200, 400, 800]

    print("Convergence Analysis - European Call:")
    print("-" * 80)
    print(f"{'Steps':<10} {'Binomial':<15} {'Error':<15} {'Trinomial':<15} {'Error':<15}")
    print("-" * 80)

    for n in step_sizes:
        # Binomial
        bin_pricer = BinomialTreePricer(
            spot=S0, strike=K, maturity=T,
            volatility=sigma, rate=r, dividend=q,
            option_type='call', exercise_type='european',
            n_steps=n
        )
        bin_price = bin_pricer.price()
        bin_error = abs(bin_price - bs_call)

        # Trinomial
        tri_pricer = TrinomialTreePricer(
            spot=S0, strike=K, maturity=T,
            volatility=sigma, rate=r, dividend=q,
            option_type='call', exercise_type='european',
            n_steps=n
        )
        tri_price = tri_pricer.price()
        tri_error = abs(tri_price - bs_call)

        print(f"{n:<10} ${bin_price:<14.6f} ${bin_error:<14.6f} "
              f"${tri_price:<14.6f} ${tri_error:<14.6f}")

    print()

    # Show convergence rate
    print("Convergence Rate Analysis:")
    print("-" * 40)
    print("Error should decrease as O(1/N) for binomial trees")
    print("Both methods converge to Black-Scholes as N → ∞")
    print()


def example_4_trinomial_comparison():
    """Example 4: Compare binomial vs trinomial trees."""
    print("=" * 80)
    print("EXAMPLE 4: Binomial vs Trinomial Tree Comparison")
    print("=" * 80)
    print()

    # Parameters
    S0 = 100.0
    K = 105.0
    T = 0.5
    sigma = 0.30  # Higher volatility
    r = 0.04
    q = 0.01
    n_steps = 300

    print(f"Parameters: S0=${S0}, K=${K}, T={T}yr, σ={sigma:.0%}, r={r:.0%}, q={q:.0%}")
    print(f"Number of steps: {n_steps}")
    print()

    # Binomial tree
    print("Binomial Tree (CRR):")
    print("-" * 40)
    start = time.perf_counter()
    bin_pricer = BinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='put', exercise_type='american',
        n_steps=n_steps
    )
    bin_price = bin_pricer.price()
    bin_time = (time.perf_counter() - start) * 1000

    bin_greeks = bin_pricer.get_greeks()

    print(f"  Price: ${bin_price:.4f}")
    print(f"  Time:  {bin_time:.2f} ms")
    print(f"  Delta: {bin_greeks.delta:.4f}")
    print(f"  Gamma: {bin_greeks.gamma:.4f}")
    print()

    # Trinomial tree
    print("Trinomial Tree:")
    print("-" * 40)
    start = time.perf_counter()
    tri_pricer = TrinomialTreePricer(
        spot=S0, strike=K, maturity=T,
        volatility=sigma, rate=r, dividend=q,
        option_type='put', exercise_type='american',
        n_steps=n_steps
    )
    tri_price = tri_pricer.price()
    tri_time = (time.perf_counter() - start) * 1000

    tri_greeks = tri_pricer.get_greeks()

    print(f"  Price: ${tri_price:.4f}")
    print(f"  Time:  {tri_time:.2f} ms")
    print(f"  Delta: {tri_greeks.delta:.4f}")
    print(f"  Gamma: {tri_greeks.gamma:.4f}")
    print()

    # Comparison
    print("Comparison:")
    print("-" * 40)
    price_diff = abs(bin_price - tri_price)
    print(f"  Price Difference: ${price_diff:.4f} ({price_diff/bin_price:.2%})")
    print(f"  Binomial Time:    {bin_time:.2f} ms")
    print(f"  Trinomial Time:   {tri_time:.2f} ms")
    print(f"  Speed Ratio:      {tri_time/bin_time:.2f}x")
    print()

    print("Analysis:")
    print("  - Both methods produce very similar prices")
    print("  - Binomial is generally faster for same number of steps")
    print("  - Trinomial can be more stable for high volatility")
    print("  - Choice depends on specific use case and accuracy requirements")
    print()


def example_5_performance_benchmark():
    """Example 5: Performance benchmarking across different tree sizes."""
    print("=" * 80)
    print("EXAMPLE 5: Performance Benchmark")
    print("=" * 80)
    print()

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    sigma = 0.20
    r = 0.05
    q = 0.02

    step_sizes = [50, 100, 200, 500, 1000]

    print("Binomial Tree Performance (American Put):")
    print("-" * 80)
    print(f"{'Steps':<10} {'Time (ms)':<15} {'Price':<15} {'Nodes':<15}")
    print("-" * 80)

    for n in step_sizes:
        pricer = BinomialTreePricer(
            spot=S0, strike=K, maturity=T,
            volatility=sigma, rate=r, dividend=q,
            option_type='put', exercise_type='american',
            n_steps=n
        )

        start = time.perf_counter()
        price = pricer.price()
        elapsed = (time.perf_counter() - start) * 1000

        # Number of nodes in tree
        nodes = (n + 1) * (n + 2) // 2

        print(f"{n:<10} {elapsed:<15.2f} ${price:<14.4f} {nodes:<15,}")

    print()

    print("Performance Notes:")
    print("  - Computation time scales as O(N²)")
    print("  - Memory usage scales as O(N²) for full tree storage")
    print("  - For N > 1000, consider memory-efficient implementation")
    print()


def example_6_greeks_analysis():
    """Example 6: Detailed Greeks analysis and sensitivity."""
    print("=" * 80)
    print("EXAMPLE 6: Greeks Analysis and Sensitivity")
    print("=" * 80)
    print()

    # Base parameters
    S0 = 100.0
    K = 100.0
    T = 0.25  # 3 months
    sigma = 0.25
    r = 0.05
    q = 0.02
    n_steps = 200

    print(f"Base Parameters: S0=${S0}, K=${K}, T={T}yr, σ={sigma:.0%}, r={r:.0%}, q={q:.0%}")
    print()

    # Analyze Greeks for different moneyness levels
    spot_prices = [80, 90, 100, 110, 120]

    print("Call Option Greeks vs Spot Price:")
    print("-" * 80)
    print(f"{'Spot':<10} {'Price':<12} {'Delta':<10} {'Gamma':<10} {'Vega':<10} {'Theta':<10}")
    print("-" * 80)

    for S in spot_prices:
        pricer = BinomialTreePricer(
            spot=S, strike=K, maturity=T,
            volatility=sigma, rate=r, dividend=q,
            option_type='call', exercise_type='european',
            n_steps=n_steps
        )
        price = pricer.price()
        greeks = pricer.get_greeks()

        print(f"${S:<9.0f} ${price:<11.4f} {greeks.delta:<9.4f} "
              f"{greeks.gamma:<9.4f} {greeks.vega:<9.4f} {greeks.theta:<9.4f}")

    print()

    print("Greek Interpretation:")
    print("  Delta:  Hedge ratio (shares per option), ranges [0,1] for calls")
    print("  Gamma:  Rate of change of delta, highest at-the-money")
    print("  Vega:   Sensitivity to volatility, highest at-the-money")
    print("  Theta:  Time decay, negative for long options")
    print()


def main():
    """Run all examples."""
    examples = [
        ("Basic Pricing", example_1_basic_pricing),
        ("American Options", example_2_american_options),
        ("Convergence Study", example_3_convergence_study),
        ("Trinomial Comparison", example_4_trinomial_comparison),
        ("Performance Benchmark", example_5_performance_benchmark),
        ("Greeks Analysis", example_6_greeks_analysis),
    ]

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "LATTICE OPTION PRICING EXAMPLES" + " " * 27 + "║")
    print("║" + " " * 15 + "Binomial & Trinomial Tree Models" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    for i, (name, func) in enumerate(examples, 1):
        print(f"\nRunning Example {i}/{len(examples)}: {name}...")
        print()
        func()
        print("\n")

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Binomial trees: Fast, reliable, industry standard")
    print("  - Trinomial trees: More stable for high volatility")
    print("  - Both converge to Black-Scholes for European options")
    print("  - American options priced via backward induction")
    print("  - Greeks calculated via finite differences")
    print("=" * 80)


if __name__ == "__main__":
    main()
