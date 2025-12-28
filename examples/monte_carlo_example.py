#!/usr/bin/env python3
"""
Monte Carlo Option Pricing - Usage Examples

This script demonstrates the Monte Carlo pricing engine with various
configurations and use cases.

Run: python examples/monte_carlo_example.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pricing.monte_carlo import MonteCarloEngine, MCConfig
from src.pricing.black_scholes import BSParameters, BlackScholesEngine


def example_1_basic_european():
    """Example 1: Basic European option pricing."""
    print("=" * 80)
    print("Example 1: Basic European Option Pricing")
    print("=" * 80)

    # Configure Monte Carlo simulation
    config = MCConfig(
        n_paths=100000,
        n_steps=252,
        antithetic=True,
        control_variate=True,
        seed=42
    )
    engine = MonteCarloEngine(config)

    # Define option parameters
    params = BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.02
    )

    print("\nParameters:")
    print(f"  Spot Price:       ${params.spot:.2f}")
    print(f"  Strike Price:     ${params.strike:.2f}")
    print(f"  Time to Maturity: {params.maturity:.2f} years")
    print(f"  Volatility:       {params.volatility:.1%}")
    print(f"  Risk-Free Rate:   {params.rate:.1%}")
    print(f"  Dividend Yield:   {params.dividend:.1%}")

    # Price call option
    print("\nPricing Call Option...")
    start = time.time()
    call_price, call_ci = engine.price_european(params, 'call')
    call_time = time.time() - start

    # Price put option
    print("Pricing Put Option...")
    start = time.time()
    put_price, put_ci = engine.price_european(params, 'put')
    put_time = time.time() - start

    # Compare to Black-Scholes
    bs_call = BlackScholesEngine.price_call(params)
    bs_put = BlackScholesEngine.price_put(params)

    print("\nResults:")
    print(f"  Call (MC):  ${call_price:.6f} ± ${call_ci:.6f}  ({call_time:.2f}s)")
    print(f"  Call (BS):  ${bs_call:.6f}")
    print(f"  Error:      {abs(call_price - bs_call) / bs_call * 100:.3f}%")
    print()
    print(f"  Put (MC):   ${put_price:.6f} ± ${put_ci:.6f}  ({put_time:.2f}s)")
    print(f"  Put (BS):   ${bs_put:.6f}")
    print(f"  Error:      {abs(put_price - bs_put) / bs_put * 100:.3f}%")


def example_2_american_option():
    """Example 2: American option with early exercise."""
    print("\n" + "=" * 80)
    print("Example 2: American Option Pricing (Early Exercise)")
    print("=" * 80)

    config = MCConfig(n_paths=50000, n_steps=100, seed=42)
    engine = MonteCarloEngine(config)

    # In-the-money put (more likely to be exercised early)
    params = BSParameters(
        spot=80.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.3,
        rate=0.05,
        dividend=0.0  # No dividend makes early exercise more valuable
    )

    print("\nParameters (Deep ITM Put):")
    print(f"  Spot:   ${params.spot:.2f}")
    print(f"  Strike: ${params.strike:.2f}")
    print(f"  Moneyness: {params.spot / params.strike:.2%}")

    # Price European put
    print("\nPricing European Put...")
    start = time.time()
    european_price, ci = engine.price_european(params, 'put')
    european_time = time.time() - start

    # Price American put
    print("Pricing American Put (Longstaff-Schwartz)...")
    start = time.time()
    american_price = engine.price_american_lsm(params, 'put')
    american_time = time.time() - start

    # Calculate early exercise premium
    premium = american_price - european_price
    premium_pct = premium / european_price * 100

    print("\nResults:")
    print(f"  European Put: ${european_price:.6f} ± ${ci:.6f}  ({european_time:.2f}s)")
    print(f"  American Put: ${american_price:.6f}  ({american_time:.2f}s)")
    print(f"  Early Exercise Premium: ${premium:.6f} ({premium_pct:.2f}%)")

    # Intrinsic value check
    intrinsic = max(params.strike - params.spot, 0)
    print(f"  Intrinsic Value: ${intrinsic:.6f}")
    print(f"  Time Value: ${american_price - intrinsic:.6f}")


def example_3_variance_reduction():
    """Example 3: Compare variance reduction techniques."""
    print("\n" + "=" * 80)
    print("Example 3: Variance Reduction Techniques")
    print("=" * 80)

    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0,
        volatility=0.2, rate=0.05, dividend=0.02
    )

    configs = [
        ("No VR", MCConfig(n_paths=50000, antithetic=False, control_variate=False, seed=42)),
        ("Antithetic Only", MCConfig(n_paths=50000, antithetic=True, control_variate=False, seed=42)),
        ("Control Only", MCConfig(n_paths=50000, antithetic=False, control_variate=True, seed=42)),
        ("Both VR", MCConfig(n_paths=50000, antithetic=True, control_variate=True, seed=42)),
    ]

    print("\nComparing Variance Reduction Techniques:")
    print(f"{'Configuration':<20} {'Price':>12} {'CI':>10} {'Reduction':>12}")
    print("-" * 60)

    baseline_ci = None
    for name, config in configs:
        engine = MonteCarloEngine(config)
        price, ci = engine.price_european(params, 'call')

        if baseline_ci is None:
            baseline_ci = ci
            reduction = "baseline"
        else:
            reduction_pct = (baseline_ci - ci) / baseline_ci * 100
            reduction = f"{reduction_pct:>10.1f}%"

        print(f"{name:<20} ${price:>10.6f}  ${ci:>8.6f}  {reduction:>12}")


def example_4_sensitivity_analysis():
    """Example 4: Volatility sensitivity (Vega approximation)."""
    print("\n" + "=" * 80)
    print("Example 4: Volatility Sensitivity Analysis")
    print("=" * 80)

    config = MCConfig(n_paths=50000, n_steps=100, seed=42)
    engine = MonteCarloEngine(config)

    base_params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0,
        volatility=0.2, rate=0.05, dividend=0.02
    )

    # Compute prices at different volatilities
    volatilities = [0.15, 0.175, 0.2, 0.225, 0.25]
    prices = []

    print("\nVolatility Analysis (Call Option):")
    print(f"{'Volatility':>12} {'Price':>12} {'BS Price':>12} {'Error':>10}")
    print("-" * 50)

    for vol in volatilities:
        params = BSParameters(
            spot=base_params.spot,
            strike=base_params.strike,
            maturity=base_params.maturity,
            volatility=vol,
            rate=base_params.rate,
            dividend=base_params.dividend
        )

        mc_price, _ = engine.price_european(params, 'call')
        bs_price = BlackScholesEngine.price_call(params)
        error = abs(mc_price - bs_price) / bs_price * 100

        prices.append(mc_price)

        print(f"{vol:>11.1%} ${mc_price:>10.6f} ${bs_price:>10.6f} {error:>9.3f}%")

    # Approximate Vega (numerical derivative)
    dvol = volatilities[1] - volatilities[0]
    vega_approx = (prices[1] - prices[0]) / dvol
    print(f"\nApproximate Vega: ${vega_approx:.4f} per 1% volatility change")

    # Compare to analytical Vega
    greeks = BlackScholesEngine.calculate_greeks(base_params, 'call')
    print(f"Analytical Vega:  ${greeks.vega:.4f} per 1% volatility change")


def example_5_performance_scaling():
    """Example 5: Performance scaling with number of paths."""
    print("\n" + "=" * 80)
    print("Example 5: Performance Scaling")
    print("=" * 80)

    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0,
        volatility=0.2, rate=0.05, dividend=0.02
    )

    path_counts = [10000, 25000, 50000, 100000, 200000]

    print("\nPerformance vs. Number of Paths:")
    print(f"{'N Paths':>10} {'Time (s)':>12} {'Paths/sec':>12} {'CI':>10}")
    print("-" * 50)

    for n_paths in path_counts:
        config = MCConfig(n_paths=n_paths, n_steps=100, seed=42)
        engine = MonteCarloEngine(config)

        # Warm-up
        _, _ = engine.price_european(params, 'call')

        # Benchmark
        start = time.time()
        price, ci = engine.price_european(params, 'call')
        elapsed = time.time() - start

        paths_per_sec = n_paths / elapsed

        print(f"{n_paths:>10,} {elapsed:>11.3f}s {paths_per_sec:>11,.0f} ${ci:>8.6f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Monte Carlo Option Pricing Examples" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")

    examples = [
        example_1_basic_european,
        example_2_american_option,
        example_3_variance_reduction,
        example_4_sensitivity_analysis,
        example_5_performance_scaling,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in Example {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
