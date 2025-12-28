"""
Exotic Options Pricing - Quick Start Guide
===========================================

This script demonstrates the usage of all exotic option types
implemented in the pricing module.

Run this file to see examples of:
1. Asian options (arithmetic and geometric)
2. Barrier options (all 8 types)
3. Lookback options (fixed and floating strike)
4. Digital options (cash and asset-or-nothing)
"""

from src.pricing.black_scholes import BSParameters, BlackScholesEngine
from src.pricing.exotic import (
    ExoticParameters,
    AsianType,
    StrikeType,
    BarrierType,
    AsianOptionPricer,
    BarrierOptionPricer,
    LookbackOptionPricer,
    DigitalOptionPricer,
    price_exotic_option
)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def example_asian_options():
    """Demonstrate Asian option pricing."""
    print_section("1. ASIAN OPTIONS")

    # Setup parameters
    base_params = BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.20,
        rate=0.05,
        dividend=0.02
    )

    asian_params = ExoticParameters(
        base_params=base_params,
        n_observations=252  # Daily observations for 1 year
    )

    print("\nParameters:")
    print(f"  Spot:        ${base_params.spot}")
    print(f"  Strike:      ${base_params.strike}")
    print(f"  Maturity:    {base_params.maturity} years")
    print(f"  Volatility:  {base_params.volatility:.1%}")
    print(f"  Rate:        {base_params.rate:.1%}")
    print(f"  Dividend:    {base_params.dividend:.1%}")
    print(f"  Observations: {asian_params.n_observations}")

    # Geometric Asian (analytical - fast)
    print("\n--- Geometric Asian (Analytical) ---")
    geom_call = AsianOptionPricer.price_geometric_asian(
        asian_params, 'call', StrikeType.FIXED
    )
    geom_put = AsianOptionPricer.price_geometric_asian(
        asian_params, 'put', StrikeType.FIXED
    )
    print(f"Call: ${geom_call:.6f}")
    print(f"Put:  ${geom_put:.6f}")

    # Arithmetic Asian (Monte Carlo)
    print("\n--- Arithmetic Asian (Monte Carlo with Control Variate) ---")
    arith_call, ci_call = AsianOptionPricer.price_arithmetic_asian_mc(
        asian_params, 'call', StrikeType.FIXED,
        n_paths=50000,
        use_control_variate=True
    )
    arith_put, ci_put = AsianOptionPricer.price_arithmetic_asian_mc(
        asian_params, 'put', StrikeType.FIXED,
        n_paths=50000,
        use_control_variate=True
    )
    print(f"Call: ${arith_call:.6f} ± ${ci_call:.6f}")
    print(f"Put:  ${arith_put:.6f} ± ${ci_put:.6f}")

    # Verify AM-GM inequality
    print("\n--- AM-GM Inequality Check ---")
    print(f"Arithmetic Call ({arith_call:.6f}) >= Geometric Call ({geom_call:.6f}): {arith_call >= geom_call}")

    # Vanilla for comparison
    vanilla_call = BlackScholesEngine.price_call(base_params)
    vanilla_put = BlackScholesEngine.price_put(base_params)
    print(f"\nVanilla European Call: ${vanilla_call:.6f}")
    print(f"Asian options are cheaper due to averaging effect")


def example_barrier_options():
    """Demonstrate barrier option pricing."""
    print_section("2. BARRIER OPTIONS")

    base_params = BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.25,
        rate=0.05,
        dividend=0.02
    )

    # Up barrier (barrier above spot)
    up_barrier_params = ExoticParameters(
        base_params=base_params,
        barrier=120.0,
        rebate=0.0
    )

    # Down barrier (barrier below spot)
    down_barrier_params = ExoticParameters(
        base_params=base_params,
        barrier=80.0,
        rebate=5.0  # Rebate if barrier hit
    )

    print("\nUp Barrier Options (H = $120):")
    print("-" * 40)

    # Up-and-Out Call
    uoc = BarrierOptionPricer.price_barrier_analytical(
        up_barrier_params, 'call', BarrierType.UP_AND_OUT
    )
    print(f"Up-and-Out Call:  ${uoc:.6f}")

    # Up-and-In Call
    uic = BarrierOptionPricer.price_barrier_analytical(
        up_barrier_params, 'call', BarrierType.UP_AND_IN
    )
    print(f"Up-and-In Call:   ${uic:.6f}")

    # Verify parity
    vanilla = BlackScholesEngine.price_call(base_params)
    print(f"\nVanilla Call:     ${vanilla:.6f}")
    print(f"Sum (UOC + UIC):  ${uoc + uic:.6f}")
    print(f"Parity Error:     ${abs((uoc + uic) - vanilla):.2e}")

    print("\nDown Barrier Options (H = $80, Rebate = $5):")
    print("-" * 40)

    # Down-and-Out Put
    dop = BarrierOptionPricer.price_barrier_analytical(
        down_barrier_params, 'put', BarrierType.DOWN_AND_OUT
    )
    print(f"Down-and-Out Put: ${dop:.6f}")

    # Down-and-In Put
    dip = BarrierOptionPricer.price_barrier_analytical(
        down_barrier_params, 'put', BarrierType.DOWN_AND_IN
    )
    print(f"Down-and-In Put:  ${dip:.6f}")

    vanilla_put = BlackScholesEngine.price_put(base_params)
    print(f"\nVanilla Put:      ${vanilla_put:.6f}")
    print(f"Sum (DOP + DIP):  ${dop + dip:.6f}")
    print("Note: Rebate adds value to knock-out option")


def example_lookback_options():
    """Demonstrate lookback option pricing."""
    print_section("3. LOOKBACK OPTIONS")

    base_params = BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.25,
        rate=0.05,
        dividend=0.02
    )

    lookback_params = ExoticParameters(
        base_params=base_params,
        n_observations=252
    )

    print("\n--- Floating Strike Lookback (Analytical) ---")
    print("Payoff: Call = S_T - min(S), Put = max(S) - S_T")
    print("These options ALWAYS finish in-the-money!\n")

    floating_call = LookbackOptionPricer.price_floating_strike_analytical(
        base_params, 'call'
    )
    floating_put = LookbackOptionPricer.price_floating_strike_analytical(
        base_params, 'put'
    )

    print(f"Floating Strike Call: ${floating_call:.6f}")
    print(f"Floating Strike Put:  ${floating_put:.6f}")

    print("\n--- Fixed Strike Lookback (Monte Carlo) ---")
    print("Payoff: Call = max(max(S) - K, 0), Put = max(K - min(S), 0)\n")

    fixed_call, ci_call = LookbackOptionPricer.price_lookback_mc(
        lookback_params, 'call', StrikeType.FIXED,
        n_paths=50000
    )
    fixed_put, ci_put = LookbackOptionPricer.price_lookback_mc(
        lookback_params, 'put', StrikeType.FIXED,
        n_paths=50000
    )

    print(f"Fixed Strike Call: ${fixed_call:.6f} ± ${ci_call:.6f}")
    print(f"Fixed Strike Put:  ${fixed_put:.6f} ± ${ci_put:.6f}")

    # Comparison to vanilla
    vanilla_call = BlackScholesEngine.price_call(base_params)
    print(f"\nVanilla European Call: ${vanilla_call:.6f}")
    print(f"Lookback uses max(S) instead of S_T, hence more valuable")


def example_digital_options():
    """Demonstrate digital option pricing."""
    print_section("4. DIGITAL (BINARY) OPTIONS")

    base_params = BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.25,
        rate=0.05,
        dividend=0.02
    )

    print("\n--- Cash-or-Nothing Options ---")
    print("Pays fixed amount if ITM, zero otherwise\n")

    cash_call_1 = DigitalOptionPricer.price_cash_or_nothing(
        base_params, 'call', payout=1.0
    )
    cash_put_1 = DigitalOptionPricer.price_cash_or_nothing(
        base_params, 'put', payout=1.0
    )

    print(f"Cash-or-Nothing Call ($1):  ${cash_call_1:.6f}")
    print(f"Cash-or-Nothing Put ($1):   ${cash_put_1:.6f}")

    # Probability interpretation
    discount = 1.0 / (1.0 + base_params.rate) ** base_params.maturity
    prob_call = cash_call_1 / discount
    prob_put = cash_put_1 / discount

    print(f"\nRisk-neutral probabilities:")
    print(f"  P(S_T > K): {prob_call:.4f}")
    print(f"  P(S_T < K): {prob_put:.4f}")
    print(f"  Sum:        {prob_call + prob_put:.4f} (should be ~1.0)")

    print("\n--- Asset-or-Nothing Options ---")
    print("Pays S_T if ITM, zero otherwise\n")

    asset_call = DigitalOptionPricer.price_asset_or_nothing(base_params, 'call')
    asset_put = DigitalOptionPricer.price_asset_or_nothing(base_params, 'put')

    print(f"Asset-or-Nothing Call: ${asset_call:.6f}")
    print(f"Asset-or-Nothing Put:  ${asset_put:.6f}")

    print("\n--- Vanilla Option Decomposition ---")
    print("Vanilla = Asset-or-Nothing - K × Cash-or-Nothing\n")

    vanilla_call = BlackScholesEngine.price_call(base_params)
    reconstructed = asset_call - base_params.strike * cash_call_1

    print(f"Vanilla Call (direct):       ${vanilla_call:.6f}")
    print(f"Vanilla Call (reconstructed): ${reconstructed:.6f}")
    print(f"Reconstruction Error:         ${abs(vanilla_call - reconstructed):.2e}")

    print("\n--- Digital Greeks ---")
    greeks = DigitalOptionPricer.calculate_digital_greeks(
        base_params, 'call', 'cash', payout=100.0
    )

    print(f"Cash-or-Nothing Call ($100) Greeks:")
    print(f"  Delta: {greeks['delta']:>10.6f}")
    print(f"  Gamma: {greeks['gamma']:>10.6f}")
    print(f"  Vega:  {greeks['vega']:>10.6f}")
    print(f"  Theta: {greeks['theta']:>10.6f}")
    print(f"  Rho:   {greeks['rho']:>10.6f}")
    print("\nWarning: Greeks can be discontinuous near strike!")


def example_unified_interface():
    """Demonstrate unified pricing interface."""
    print_section("5. UNIFIED PRICING INTERFACE")

    base_params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0,
        volatility=0.25, rate=0.05, dividend=0.02
    )

    print("\nThe unified interface allows pricing all exotic options")
    print("through a single function call:\n")

    # Asian
    asian_params = ExoticParameters(base_params=base_params, n_observations=252)
    price, ci = price_exotic_option(
        'asian', asian_params, 'call',
        asian_type=AsianType.GEOMETRIC
    )
    print(f"1. Geometric Asian Call: ${price:.6f} (CI: {ci})")

    # Barrier
    barrier_params = ExoticParameters(base_params=base_params, barrier=120.0)
    price, ci = price_exotic_option(
        'barrier', barrier_params, 'call',
        barrier_type=BarrierType.UP_AND_OUT
    )
    print(f"2. Up-and-Out Call:      ${price:.6f} (CI: {ci})")

    # Lookback
    lookback_params = ExoticParameters(base_params=base_params, n_observations=252)
    price, ci = price_exotic_option(
        'lookback', lookback_params, 'call',
        strike_type=StrikeType.FLOATING
    )
    print(f"3. Floating Lookback:    ${price:.6f} (CI: {ci})")

    # Digital
    digital_params = ExoticParameters(base_params=base_params)
    price, ci = price_exotic_option(
        'digital', digital_params, 'call',
        digital_type='cash', payout=100.0
    )
    print(f"4. Cash Digital ($100):  ${price:.6f} (CI: {ci})")

    print("\nNote: CI is None for analytical methods, float for Monte Carlo")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" EXOTIC OPTIONS PRICING - QUICK START GUIDE")
    print("=" * 80)
    print("\nThis guide demonstrates all exotic option types implemented")
    print("in the pricing module with practical examples.\n")

    example_asian_options()
    example_barrier_options()
    example_lookback_options()
    example_digital_options()
    example_unified_interface()

    print("\n" + "=" * 80)
    print(" END OF QUICK START GUIDE")
    print("=" * 80)
    print("\nFor more details, see:")
    print("  - Documentation: docs/EXOTIC_OPTIONS.md")
    print("  - Source code:   src/pricing/exotic.py")
    print("  - Unit tests:    tests/test_exotic.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
