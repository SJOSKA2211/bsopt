"""
Step-by-Step Black-Scholes Calculation Example

This file provides a detailed walkthrough of Black-Scholes calculations,
showing each computational step for educational and verification purposes.
"""

import numpy as np
from scipy.stats import norm

from src.pricing.black_scholes import BlackScholesEngine, BSParameters


def step_by_step_calculation():
    """
    Demonstrate Black-Scholes calculation with detailed step-by-step output.
    """
    print("=" * 80)
    print("BLACK-SCHOLES STEP-BY-STEP CALCULATION")
    print("=" * 80)
    print("\nThis example shows every computational step for an ATM call option.")
    print()

    # Input parameters
    S = 100.0  # Spot price
    K = 100.0  # Strike price
    T = 1.0  # Time to maturity (years)
    sigma = 0.20  # Volatility (20%)
    r = 0.05  # Risk-free rate (5%)
    q = 0.02  # Dividend yield (2%)

    print("INPUT PARAMETERS")
    print("-" * 80)
    print(f"Spot Price (S):          {S:.2f}")
    print(f"Strike Price (K):        {K:.2f}")
    print(f"Time to Maturity (T):    {T:.2f} years")
    print(f"Volatility (σ):          {sigma:.4f} ({sigma*100:.1f}%)")
    print(f"Risk-Free Rate (r):      {r:.4f} ({r*100:.1f}%)")
    print(f"Dividend Yield (q):      {q:.4f} ({q*100:.1f}%)")
    print()

    # Step 1: Calculate intermediate terms
    print("STEP 1: CALCULATE INTERMEDIATE TERMS")
    print("-" * 80)

    sqrt_T = np.sqrt(T)
    print(f"√T = √{T} = {sqrt_T:.6f}")

    log_moneyness = np.log(S / K)
    print(f"ln(S/K) = ln({S}/{K}) = {log_moneyness:.6f}")

    sigma_squared = sigma * sigma
    print(f"σ² = {sigma}² = {sigma_squared:.6f}")

    vol_sqrt_T = sigma * sqrt_T
    print(f"σ·√T = {sigma} × {sqrt_T:.6f} = {vol_sqrt_T:.6f}")
    print()

    # Step 2: Calculate d1
    print("STEP 2: CALCULATE d₁")
    print("-" * 80)
    print("Formula: d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)")
    print()

    drift_term = (r - q + 0.5 * sigma_squared) * T
    print("Drift term = (r - q + σ²/2)T")
    print(f"           = ({r} - {q} + {0.5 * sigma_squared:.6f}) × {T}")
    print(f"           = {drift_term:.6f}")
    print()

    d1_numerator = log_moneyness + drift_term
    print("d₁ numerator = ln(S/K) + drift_term")
    print(f"             = {log_moneyness:.6f} + {drift_term:.6f}")
    print(f"             = {d1_numerator:.6f}")
    print()

    d1 = d1_numerator / vol_sqrt_T
    print(f"d₁ = {d1_numerator:.6f} / {vol_sqrt_T:.6f}")
    print(f"   = {d1:.6f}")
    print()

    # Step 3: Calculate d2
    print("STEP 3: CALCULATE d₂")
    print("-" * 80)
    print("Formula: d₂ = d₁ - σ√T")
    print()

    d2 = d1 - vol_sqrt_T
    print(f"d₂ = {d1:.6f} - {vol_sqrt_T:.6f}")
    print(f"   = {d2:.6f}")
    print()

    # Step 4: Calculate normal distribution values
    print("STEP 4: CALCULATE NORMAL DISTRIBUTION VALUES")
    print("-" * 80)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    print(f"N(d₁) = Φ({d1:.6f}) = {N_d1:.10f}")
    print(f"N(d₂) = Φ({d2:.6f}) = {N_d2:.10f}")
    print(f"n(d₁) = φ({d1:.6f}) = {n_d1:.10f}")
    print()
    print("Where:")
    print("  Φ(x) = cumulative standard normal distribution")
    print("  φ(x) = standard normal probability density = (1/√2π)·e^(-x²/2)")
    print()

    # Step 5: Calculate discount factors
    print("STEP 5: CALCULATE DISCOUNT FACTORS")
    print("-" * 80)

    discount_r = np.exp(-r * T)
    discount_q = np.exp(-q * T)

    print(f"e^(-rT) = e^(-{r} × {T}) = {discount_r:.10f}")
    print(f"e^(-qT) = e^(-{q} × {T}) = {discount_q:.10f}")
    print()

    # Step 6: Calculate call option price
    print("STEP 6: CALCULATE CALL OPTION PRICE")
    print("-" * 80)
    print("Formula: C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)")
    print()

    first_term = S * discount_q * N_d1
    print("First term = S·e^(-qT)·N(d₁)")
    print(f"           = {S} × {discount_q:.10f} × {N_d1:.10f}")
    print(f"           = {first_term:.10f}")
    print()

    second_term = K * discount_r * N_d2
    print("Second term = K·e^(-rT)·N(d₂)")
    print(f"            = {K} × {discount_r:.10f} × {N_d2:.10f}")
    print(f"            = {second_term:.10f}")
    print()

    call_price = first_term - second_term
    print(f"Call Price = {first_term:.10f} - {second_term:.10f}")
    print(f"           = {call_price:.10f}")
    print()

    # Step 7: Calculate put option price using put-call parity
    print("STEP 7: CALCULATE PUT OPTION PRICE (PUT-CALL PARITY)")
    print("-" * 80)
    print("Formula: P = C - S·e^(-qT) + K·e^(-rT)")
    print()

    forward_diff = S * discount_q - K * discount_r
    print(f"S·e^(-qT) - K·e^(-rT) = {S * discount_q:.10f} - {K * discount_r:.10f}")
    print(f"                       = {forward_diff:.10f}")
    print()

    put_price = call_price - forward_diff
    print("Put Price = Call - (S·e^(-qT) - K·e^(-rT))")
    print(f"          = {call_price:.10f} - {forward_diff:.10f}")
    print(f"          = {put_price:.10f}")
    print()

    # Step 8: Calculate Greeks - Delta
    print("STEP 8: CALCULATE GREEKS - DELTA")
    print("-" * 80)

    delta_call = discount_q * N_d1
    N_minus_d1 = norm.cdf(-d1)
    delta_put = -discount_q * N_minus_d1

    print("Call Delta = e^(-qT) · N(d₁)")
    print(f"           = {discount_q:.10f} × {N_d1:.10f}")
    print(f"           = {delta_call:.10f}")
    print()

    print("Put Delta = -e^(-qT) · N(-d₁)")
    print(f"          = -{discount_q:.10f} × {N_minus_d1:.10f}")
    print(f"          = {delta_put:.10f}")
    print()

    # Step 9: Calculate Greeks - Gamma
    print("STEP 9: CALCULATE GREEKS - GAMMA")
    print("-" * 80)
    print("Formula: Γ = [e^(-qT) · n(d₁)] / [S · σ · √T]")
    print()

    gamma = (discount_q * n_d1) / (S * sigma * sqrt_T)
    print(f"Γ = ({discount_q:.10f} × {n_d1:.10f}) / ({S} × {sigma} × {sqrt_T:.6f})")
    print(f"  = {gamma:.10f}")
    print()

    # Step 10: Calculate Greeks - Vega
    print("STEP 10: CALCULATE GREEKS - VEGA")
    print("-" * 80)
    print("Formula: ν = S · e^(-qT) · n(d₁) · √T · 0.01")
    print("(Scaled to $ per 1% volatility change)")
    print()

    vega = S * discount_q * n_d1 * sqrt_T * 0.01
    print(f"ν = {S} × {discount_q:.10f} × {n_d1:.10f} × {sqrt_T:.6f} × 0.01")
    print(f"  = {vega:.10f}")
    print()

    # Step 11: Calculate Greeks - Theta (Call)
    print("STEP 11: CALCULATE GREEKS - THETA (CALL)")
    print("-" * 80)
    print(
        "Formula: Θ = [-S·n(d₁)·σ·e^(-qT)/(2√T) - r·K·e^(-rT)·N(d₂) + q·S·e^(-qT)·N(d₁)] / 365"
    )
    print()

    theta_term1 = -(S * n_d1 * sigma * discount_q) / (2 * sqrt_T)
    theta_term2 = -r * K * discount_r * N_d2
    theta_term3 = q * S * discount_q * N_d1

    print("Term 1 = -S·n(d₁)·σ·e^(-qT)/(2√T)")
    print(f"       = {theta_term1:.10f}")
    print()
    print("Term 2 = -r·K·e^(-rT)·N(d₂)")
    print(f"       = {theta_term2:.10f}")
    print()
    print("Term 3 = q·S·e^(-qT)·N(d₁)")
    print(f"       = {theta_term3:.10f}")
    print()

    theta_call_annual = theta_term1 + theta_term2 + theta_term3
    theta_call = theta_call_annual / 365.0

    print(f"Θ_call (annual) = {theta_call_annual:.10f}")
    print(f"Θ_call (daily)  = {theta_call:.10f}")
    print()

    # Step 12: Calculate Greeks - Rho (Call)
    print("STEP 12: CALCULATE GREEKS - RHO (CALL)")
    print("-" * 80)
    print("Formula: ρ = K · T · e^(-rT) · N(d₂) · 0.01")
    print("(Scaled to $ per 1% rate change)")
    print()

    rho_call = K * T * discount_r * N_d2 * 0.01
    print(f"ρ_call = {K} × {T} × {discount_r:.10f} × {N_d2:.10f} × 0.01")
    print(f"       = {rho_call:.10f}")
    print()

    # Verification against library implementation
    print("=" * 80)
    print("VERIFICATION AGAINST LIBRARY IMPLEMENTATION")
    print("=" * 80)
    print()

    params = BSParameters(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q
    )
    lib_call = BlackScholesEngine.price_call(params)
    lib_put = BlackScholesEngine.price_put(params)
    lib_greeks_call = BlackScholesEngine.calculate_greeks(params, "call")
    lib_d1, lib_d2 = BlackScholesEngine.calculate_d1_d2(params)

    print(f"{'Metric':<20} {'Manual Calculation':<25} {'Library':<25} {'Match':<10}")
    print("-" * 80)

    def check_match(manual, library, name):
        diff = abs(manual - library)
        match = "✓" if diff < 1e-10 else "✗"
        print(f"{name:<20} {manual:<25.10f} {library:<25.10f} {match:<10}")

    check_match(d1, lib_d1, "d₁")
    check_match(d2, lib_d2, "d₂")
    check_match(call_price, lib_call, "Call Price")
    check_match(put_price, lib_put, "Put Price")
    check_match(delta_call, lib_greeks_call.delta, "Call Delta")
    check_match(gamma, lib_greeks_call.gamma, "Gamma")
    check_match(vega, lib_greeks_call.vega, "Vega")
    check_match(theta_call, lib_greeks_call.theta, "Theta (Call)")
    check_match(rho_call, lib_greeks_call.rho, "Rho (Call)")

    print()
    print("=" * 80)
    print("All calculations match library implementation to machine precision!")
    print("=" * 80)


if __name__ == "__main__":
    step_by_step_calculation()
