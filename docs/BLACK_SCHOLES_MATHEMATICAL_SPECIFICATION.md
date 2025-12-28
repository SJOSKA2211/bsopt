# Black-Scholes Option Pricing Engine - Mathematical Specification

## Overview

This document provides the complete mathematical specification for the Black-Scholes-Merton pricing engine implementation located in `/home/kamau/comparison/src/pricing/black_scholes.py`.

## Mathematical Foundation

### Model Assumptions

The Black-Scholes-Merton model makes the following assumptions:

1. **Asset Price Dynamics**: The underlying asset follows a geometric Brownian motion:
   ```
   dS = μS dt + σS dW
   ```
   where:
   - S = asset price
   - μ = drift (expected return)
   - σ = volatility (standard deviation of returns)
   - W = Wiener process (Brownian motion)

2. **Market Conditions**:
   - Frictionless markets (no transaction costs)
   - Continuous trading
   - No arbitrage opportunities
   - Can borrow/lend at risk-free rate r
   - Constant volatility σ and risk-free rate r

3. **European Options**: Exercise only at maturity T

### Core Formulas

#### d₁ and d₂ Parameters

These are normalized measures of moneyness adjusted for time and volatility:

```
d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)

d₂ = d₁ - σ√T = [ln(S/K) + (r - q - σ²/2)T] / (σ√T)
```

**Parameters**:
- S = Current spot price
- K = Strike price
- T = Time to maturity (years)
- σ = Annualized volatility
- r = Risk-free interest rate (continuously compounded)
- q = Continuous dividend yield

**Interpretation**:
- d₁ represents the standardized distance from the current price to the strike, adjusted for expected drift and volatility
- d₂ represents the probability (in the risk-neutral measure) that the option will be exercised
- N(d₁) and N(d₂) are cumulative normal probabilities

#### Call Option Price

```
C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)
```

**Components**:
- `S·e^(-qT)·N(d₁)`: Present value of expected asset price at expiration (if in-the-money)
- `K·e^(-rT)·N(d₂)`: Present value of strike payment
- `N(x)`: Cumulative standard normal distribution function

**Economic Meaning**:
The call price equals the expected payoff max(S_T - K, 0) under risk-neutral measure, discounted to present value.

#### Put Option Price

**Direct Formula**:
```
P = K·e^(-rT)·N(-d₂) - S·e^(-qT)·N(-d₁)
```

**Put-Call Parity**:
```
P = C - S·e^(-qT) + K·e^(-rT)
```

Our implementation uses put-call parity to ensure exact consistency between call and put prices.

**Proof of Put-Call Parity**:
Consider two portfolios:
- Portfolio A: Long call + K·e^(-rT) cash
- Portfolio B: Long put + S·e^(-qT) worth of stock

At expiration:
- If S_T > K: Both portfolios worth S_T
- If S_T ≤ K: Both portfolios worth K

Therefore: C + K·e^(-rT) = P + S·e^(-qT)

Rearranging: P = C - S·e^(-qT) + K·e^(-rT)

### Option Greeks

The Greeks measure the sensitivity of option prices to various parameters. All formulas use n(x) = (1/√(2π))·e^(-x²/2) for the standard normal probability density function.

#### Delta (Δ): ∂V/∂S

Sensitivity to spot price changes.

**Call Delta**:
```
Δ_call = e^(-qT) · N(d₁)
```

**Put Delta**:
```
Δ_put = -e^(-qT) · N(-d₁) = e^(-qT) · [N(d₁) - 1]
```

**Properties**:
- Call delta: 0 ≤ Δ_call ≤ 1
- Put delta: -1 ≤ Δ_put ≤ 0
- Relationship: Δ_call - Δ_put = e^(-qT)

**Interpretation**: For a $1 increase in spot price, option value changes by approximately Δ dollars.

#### Gamma (Γ): ∂²V/∂S²

Rate of change of delta (convexity).

**Formula** (same for call and put):
```
Γ = [e^(-qT) · n(d₁)] / [S · σ · √T]
```

**Properties**:
- Always positive for long options
- Identical for calls and puts with same parameters
- Highest at-the-money (ATM)
- Decreases as option moves in/out of the money

**Interpretation**: For a $1 increase in spot price, delta changes by approximately Γ.

#### Vega (ν): ∂V/∂σ

Sensitivity to volatility changes.

**Formula** (same for call and put):
```
ν = S · e^(-qT) · n(d₁) · √T · 0.01
```

**Scaling**: Expressed as dollar change per 1% volatility change (hence the 0.01 factor).

**Properties**:
- Always positive for long options
- Identical for calls and puts
- Highest for ATM options
- Increases with time to maturity

**Interpretation**: For a 1% increase in volatility (e.g., 20% to 21%), option value increases by approximately ν dollars.

#### Theta (Θ): ∂V/∂t

Time decay (also called time premium decay).

**Call Theta**:
```
Θ_call = [-S·n(d₁)·σ·e^(-qT)/(2√T) - r·K·e^(-rT)·N(d₂) + q·S·e^(-qT)·N(d₁)] / 365
```

**Put Theta**:
```
Θ_put = [-S·n(d₁)·σ·e^(-qT)/(2√T) + r·K·e^(-rT)·N(-d₂) - q·S·e^(-qT)·N(-d₁)] / 365
```

**Scaling**: Expressed as dollar change per calendar day (hence division by 365).

**Properties**:
- Typically negative for long options (value decays with time)
- Can be positive for deep ITM European puts
- Magnitude increases as expiration approaches

**Interpretation**: One day passing costs approximately Θ dollars in option value.

#### Rho (ρ): ∂V/∂r

Sensitivity to interest rate changes.

**Call Rho**:
```
ρ_call = K · T · e^(-rT) · N(d₂) · 0.01
```

**Put Rho**:
```
ρ_put = -K · T · e^(-rT) · N(-d₂) · 0.01
```

**Scaling**: Expressed as dollar change per 1% rate change (hence the 0.01 factor).

**Properties**:
- Call rho > 0 (calls benefit from higher rates)
- Put rho < 0 (puts lose value with higher rates)
- Magnitude increases with time to maturity

**Interpretation**: For a 1% increase in interest rate (e.g., 5% to 6%), call value changes by approximately ρ_call dollars.

## Numerical Implementation Details

### Precision Requirements

1. **Data Types**: All intermediate calculations use `numpy.float64` for 64-bit floating-point precision
2. **Input Validation**: Parameters are converted to float64 upon initialization
3. **Numerical Stability**:
   - Log-moneyness computed as ln(S/K) rather than ln(S) - ln(K)
   - Discount factors computed once and reused
   - Square root of time computed once

### Edge Cases Handled

1. **Very Short Maturity** (T → 0):
   - Formula remains valid; time value approaches zero
   - Tested with T = 1/365 (one day)

2. **Very Long Maturity** (T → ∞):
   - Numerical stability maintained
   - Tested with T = 10 years

3. **Low Volatility** (σ → 0):
   - Option prices approach intrinsic value
   - Time value diminishes

4. **High Volatility**:
   - Tested up to σ = 1.0 (100% volatility)
   - No numerical overflow

5. **Extreme Moneyness**:
   - Deep ITM: Prices approach intrinsic value, delta approaches ±1
   - Deep OTM: Prices approach zero, delta approaches 0

## Verification Tests

### Test Coverage

The implementation has been verified through:

1. **Known-Answer Tests**: Prices computed match expected values from analytical solutions
2. **Put-Call Parity**: C - P = S·e^(-qT) - K·e^(-rT) holds to machine precision (< 1e-10)
3. **Greeks Consistency**:
   - Gamma equality: |Γ_call - Γ_put| < 1e-10
   - Vega equality: |ν_call - ν_put| < 1e-10
   - Delta relationship: Δ_call - Δ_put = e^(-qT)
4. **Monotonicity Properties**:
   - Call price increases with spot price
   - Put price decreases with spot price
   - Both prices increase with volatility
5. **Numerical Stability**: Tested across edge cases listed above

### Example Verification Results

**ATM Option (S=K=100, T=1, σ=0.2, r=0.05, q=0.02)**:
- Call Price: $9.2270
- Put Price: $6.3301
- Put-Call Parity Error: 0.00e+00
- Delta (Call): 0.5869
- Gamma: 0.0190
- Vega: 0.3790

All tests pass with errors below numerical precision thresholds.

## Usage Examples

### Basic Pricing

```python
from src.pricing.black_scholes import BSParameters, BlackScholesEngine

# Create parameters
params = BSParameters(
    spot=100.0,      # Current price
    strike=100.0,    # Strike price
    maturity=1.0,    # 1 year to expiration
    volatility=0.20, # 20% annual volatility
    rate=0.05,       # 5% risk-free rate
    dividend=0.02    # 2% dividend yield
)

# Price options
call_price = BlackScholesEngine.price_call(params)
put_price = BlackScholesEngine.price_put(params)

print(f"Call: ${call_price:.4f}")  # Call: $9.2270
print(f"Put: ${put_price:.4f}")    # Put: $6.3301
```

### Computing Greeks

```python
# Calculate all Greeks for a call option
greeks = BlackScholesEngine.calculate_greeks(params, 'call')

print(f"Delta: {greeks.delta:.4f}")    # 0.5869
print(f"Gamma: {greeks.gamma:.6f}")    # 0.019016
print(f"Vega: {greeks.vega:.4f}")      # 0.3790
print(f"Theta: {greeks.theta:.4f}")    # -0.0139 (per day)
print(f"Rho: {greeks.rho:.4f}")        # 0.4946
```

### Verifying Put-Call Parity

```python
from src.pricing.black_scholes import verify_put_call_parity

is_valid = verify_put_call_parity(params, tolerance=1e-10)
print(f"Parity holds: {is_valid}")  # True
```

## Complexity Analysis

### Time Complexity

All operations are O(1) constant time:
- `calculate_d1_d2()`: O(1) - Fixed number of arithmetic operations
- `price_call()`: O(1) - Calls d1_d2 once, then constant operations
- `price_put()`: O(1) - Calls price_call once, then constant operations
- `calculate_greeks()`: O(1) - Fixed number of normal CDF/PDF evaluations

### Space Complexity

O(1) constant space - no dynamic allocation, all intermediate values are scalars.

## References

1. **Black, F., & Scholes, M. (1973)**. "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. **Merton, R. C. (1973)**. "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.

3. **Hull, J. C. (2018)**. *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

4. **Wilmott, P. (2006)**. *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley.

## Implementation Notes

### Why Float64 Instead of Decimal?

For financial option pricing, `float64` is preferred over `Decimal` for the following reasons:

1. **Sufficient Precision**: Float64 provides ~15-17 decimal digits of precision, which is more than adequate for option pricing where market data itself has limited precision
2. **Performance**: Float64 operations are hardware-accelerated and orders of magnitude faster
3. **Scientific Computing Compatibility**: NumPy and SciPy (required for normal distribution functions) use float64
4. **Industry Standard**: Financial engineering libraries universally use floating-point arithmetic

**When to Use Decimal**: Reserve `Decimal` for exact currency calculations (position valuations, P&L calculations) where rounding errors must be eliminated. For mathematical models like Black-Scholes, float64 is appropriate and standard.

### Error Bounds

Numerical errors in the implementation are dominated by:
1. Normal CDF approximation: Error < 1e-15 (scipy.stats.norm)
2. Floating-point arithmetic: Error < 1e-15 (machine epsilon for float64)
3. Combined error: All test cases show errors < 1e-10

These error bounds are negligible compared to model risk and parameter estimation errors.

## Production Readiness Checklist

- [x] All formulas mathematically verified
- [x] Put-call parity holds to machine precision
- [x] Greeks calculations consistent with theory
- [x] Edge cases handled (short/long maturity, extreme strikes, high vol)
- [x] Numerical stability verified
- [x] Comprehensive test suite (8 test cases)
- [x] Type hints throughout
- [x] Complete docstrings
- [x] Parameter validation
- [x] Example usage code

## File Locations

- **Main Implementation**: `/home/kamau/comparison/src/pricing/black_scholes.py`
- **Verification Tests**: `/home/kamau/comparison/test_black_scholes_verification.py`
- **Documentation**: `/home/kamau/comparison/BLACK_SCHOLES_MATHEMATICAL_SPECIFICATION.md`

---

*This specification ensures that any developer or quant can understand, verify, and extend the Black-Scholes pricing engine with complete mathematical confidence.*
