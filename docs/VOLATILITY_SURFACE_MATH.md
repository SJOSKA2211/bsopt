# Volatility Surface Calibration - Mathematical Documentation

## Table of Contents
1. [Overview](#overview)
2. [SVI Model](#svi-model)
3. [SABR Model](#sabr-model)
4. [Calibration Theory](#calibration-theory)
5. [Arbitrage Conditions](#arbitrage-conditions)
6. [Numerical Algorithms](#numerical-algorithms)
7. [References](#references)

---

## Overview

This document provides the complete mathematical specification for volatility surface modeling using SVI and SABR models. All formulas are presented with precise computational steps suitable for implementation.

### Notation

| Symbol | Description | Type |
|--------|-------------|------|
| K | Strike price | Decimal (exact) |
| F | Forward price | Decimal (exact) |
| T | Time to maturity (years) | float64 |
| σ_impl | Implied volatility | float64 |
| w | Total variance = σ²·T | float64 |
| k | Log-moneyness = ln(K/F) | float64 |
| r | Risk-free rate | float64 |

### Precision Requirements

**Financial Calculations (Exact Precision Required):**
- Strike prices (K): `Decimal` with scale ≥ 2
- Forward prices (F): `Decimal` with scale ≥ 2
- Option prices: `Decimal` with scale ≥ 4
- Vega, Greeks: `Decimal` with scale ≥ 6

**Scientific Calculations (Approximation Acceptable):**
- Implied volatility (σ): `float64` (15-17 significant digits)
- Total variance (w): `float64`
- Model parameters: `float64`
- Optimization variables: `float64` (for performance)

**Conversion Protocol:**
```
Input: K, F as Decimal → Convert to float for volatility calculation
Volatility Model: Compute σ as float64
Output: Return σ as float, convert prices back to Decimal
```

---

## SVI Model

### 1. Model Specification

The SVI (Stochastic Volatility Inspired) model parameterizes total implied variance as a function of log-moneyness.

**Total Variance Formula:**
```
w(k) = a + b · [ρ · (k - m) + sqrt((k - m)² + σ²)]
```

**Implied Volatility:**
```
σ_impl(k, T) = sqrt(w(k) / T)
```

where:
- k = ln(K/F) is log-moneyness
- w is total variance (σ²·T)
- T > 0 is time to maturity

### 2. Parameters

**Raw SVI Parameters:**

| Parameter | Symbol | Constraint | Economic Interpretation |
|-----------|--------|------------|------------------------|
| Vertical shift | a | a + b·σ·√(1-ρ²) ≥ 0 | Minimum variance level |
| Slope | b | b ≥ 0 | Smile curvature intensity |
| Correlation | ρ | -1 < ρ < 1 | Skew direction and magnitude |
| Horizontal shift | m | unconstrained | ATM log-moneyness position |
| Variance width | σ | σ > 0 | Smile width/spread |

**Parameter Bounds for Optimization:**
```
Bounds:
  a ∈ [ε, ∞)         where ε = 1e-6
  b ∈ [ε, ∞)
  ρ ∈ (-1 + ε, 1 - ε)
  m ∈ ℝ
  σ ∈ [ε, ∞)
```

### 3. Natural Parameterization

For improved numerical stability during calibration:

**Natural Parameters:** {Δ, μ, ρ, ω, ζ}

**Conversion to Raw:**
```
Let D = sqrt(1 + ζ² - 2ρζ)

Then:
  a = Δ
  b = ω / D
  m = μ
  σ = ζ · D
  (ρ unchanged)
```

**Advantages:**
- Better conditioned Jacobian for optimization
- Parameters have more direct interpretation
- Δ: ATM total variance
- ω: ATM curvature (always positive)
- ζ: Tail decay parameter

### 4. Computational Algorithm

**Step-by-Step Calculation of σ_impl(K, F, T):**

```
Input: K (Decimal), F (Decimal), T (float), params (SVIParameters)
Output: σ_impl (float)

Step 1: Convert to float and compute log-moneyness
  K_float ← float(K)
  F_float ← float(F)
  k ← ln(K_float / F_float)

Step 2: Shift and compute discriminant
  k_shifted ← k - params.m
  discriminant ← sqrt(k_shifted² + params.σ²)

Step 3: Compute total variance
  w ← params.a + params.b · (params.ρ · k_shifted + discriminant)

Step 4: Handle edge cases
  if w < 0:
    w ← max(w, 0)  // Floor at zero (shouldn't happen with valid params)
    warn("Negative variance encountered")

Step 5: Convert to implied volatility
  σ_impl ← sqrt(w / T)

Return: σ_impl
```

**Vectorized Implementation:**
```python
def implied_volatility(k: np.ndarray, T: float, params: SVIParameters) -> np.ndarray:
    k_shifted = k - params.m
    discriminant = np.sqrt(k_shifted**2 + params.sigma**2)
    w = params.a + params.b * (params.rho * k_shifted + discriminant)
    w = np.maximum(w, 0.0)  # Ensure non-negative
    return np.sqrt(w / T)
```

### 5. Derivatives

**First Derivative (Variance Slope):**
```
dw/dk = b · [ρ + (k - m) / sqrt((k - m)² + σ²)]
```

**Second Derivative (Variance Curvature):**
```
d²w/dk² = b · σ² / [(k - m)² + σ²]^(3/2)
```

**Usage:** Required for arbitrage detection and smile dynamics.

### 6. No-Arbitrage Conditions

**Condition 1: Non-Negative Variance**
```
w(k) ≥ 0  for all k ∈ ℝ

Sufficient condition:
  a + b·σ·sqrt(1 - ρ²) ≥ 0
```

**Condition 2: Durrleman's No-Arbitrage Condition**
```
g(k) ≥ 0  for all k

where:
  g(k) = [1 - k·w'(k)/(2w(k))]² - [w'(k)]²/4 · [1/w(k) + 1/4]
```

**Computational Algorithm:**
```
Function: check_durrleman(k_grid, params)

  For each k in k_grid:
    w ← total_variance(k)
    w_prime ← variance_derivative(k)

    if w < ε:  // Avoid division by very small numbers
      w ← ε

    term1 ← (1 - k · w_prime / (2w))²
    term2 ← (w_prime)² / 4 · (1/w + 0.25)

    g[k] ← term1 - term2

  return all(g ≥ -tolerance)
```

**Tolerance:** Use `tolerance = 1e-8` for strict checks, `1e-6` for numerical stability.

---

## SABR Model

### 1. Model Specification

The SABR (Stochastic Alpha Beta Rho) model describes joint dynamics of forward price and volatility.

**Model Dynamics:**
```
dF_t = α_t · F_t^β · dW_t^F
dα_t = ν · α_t · dW_t^α

with correlation:
  E[dW_t^F · dW_t^α] = ρ dt
```

**Parameters:**

| Parameter | Symbol | Range | Interpretation |
|-----------|--------|-------|----------------|
| Initial vol | α | α > 0 | Volatility level at t=0 |
| CEV exponent | β | 0 ≤ β ≤ 1 | Process type (0=normal, 1=lognormal) |
| Correlation | ρ | -1 < ρ < 1 | Price-vol correlation |
| Vol-of-vol | ν | ν ≥ 0 | Volatility variance |

**Special Cases:**
- β = 0: Normal (absolute) model
- β = 0.5: Square-root (CIR-like) model
- β = 1: Lognormal (Black-Scholes) model

### 2. Hagan's Asymptotic Formula

**Implied Volatility (Full Formula):**

```
σ_impl(K, F, T) = [α / (F·K)^((1-β)/2)] · [z / x(z)] · [1 + U·T]

where:
  z = (ν/α) · (F·K)^((1-β)/2) · ln(F/K)

  x(z) = ln[(sqrt(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]

  U = [(1-β)²/24] · [α² / (F·K)^(1-β)]
      + [ρ·β·ν·α / 4] · [1 / (F·K)^((1-β)/2)]
      + [(2 - 3ρ²) / 24] · ν²
```

### 3. Computational Algorithm

**Case 1: ATM (K ≈ F)**

When |ln(F/K)| < threshold (typically 1e-6):

```
σ_ATM = [α / F^(1-β)] · [1 + U_ATM · T]

where:
  U_ATM = [(1-β)²/24] · [α² / F^(2(1-β))]
          + [ρ·β·ν·α / 4] · [1 / F^(1-β)]
          + [(2 - 3ρ²) / 24] · ν²
```

**Algorithm:**
```
Function: sabr_atm_volatility(F, T, params)

  if β == 1.0:
    base_vol ← α
    U_ATM ← (2 - 3ρ²) / 24 · ν²
  else:
    F_power ← F^(1 - β)
    base_vol ← α / F_power

    term1 ← (1 - β)² / 24 · α² / F^(2(1-β))
    term2 ← ρ·β·ν·α / 4 · (1 / F_power)
    term3 ← (2 - 3ρ²) / 24 · ν²

    U_ATM ← term1 + term2 + term3

  correction ← 1 + U_ATM · T

  return base_vol · correction
```

**Case 2: OTM/ITM (K ≠ F)**

```
Function: sabr_otm_volatility(K, F, T, params)

  Step 1: Geometric average
    FK ← F · K
    FK_mid ← sqrt(FK)

  Step 2: Log-moneyness
    log_FK ← ln(F / K)

  Step 3: Compute z parameter
    if β == 1.0:
      z ← (ν / α) · log_FK
    else:
      z ← (ν / α) · FK_mid^(1 - β) · log_FK

  Step 4: Compute x(z) with stability
    x_z ← compute_x_function(z, ρ)

  Step 5: Base volatility
    if β == 1.0:
      base_vol ← α
    else:
      base_vol ← α / FK_mid^(1 - β)

  Step 6: z/x(z) ratio
    if |z| < 1e-7:
      z_over_x ← 1.0 - ρ·z/2  // Taylor expansion
    else:
      z_over_x ← z / x_z

  Step 7: Time correction U
    if β == 1.0:
      U ← (2 - 3ρ²) / 24 · ν²
    else:
      term1 ← (1-β)² / 24 · α² / FK_mid^(2(1-β))
      term2 ← ρ·β·ν·α / 4 / FK_mid^(1-β)
      term3 ← (2 - 3ρ²) / 24 · ν²
      U ← term1 + term2 + term3

  Step 8: Final volatility
    correction ← 1 + U·T
    σ_impl ← base_vol · z_over_x · correction

  return σ_impl
```

### 4. x(z) Function Computation

**Mathematical Definition:**
```
x(z) = ln[(sqrt(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
```

**Numerical Stable Implementation:**

```
Function: compute_x(z, ρ)

  // For small z, use Taylor expansion around z=0
  if |z| < 1e-7:
    return z · (1 - ρ)

  // Check discriminant
  discriminant ← 1 - 2ρz + z²

  if discriminant < 0:
    warn("Negative discriminant in x(z)")
    discriminant ← |discriminant|

  sqrt_disc ← sqrt(discriminant)
  numerator ← sqrt_disc + z - ρ
  denominator ← 1 - ρ

  // Handle ρ ≈ 1 case (L'Hôpital's rule)
  if |denominator| < 1e-10:
    return z

  return ln(numerator / denominator)
```

**Taylor Expansion for Small z:**
```
x(z) ≈ z(1 - ρ) + z³·ρ(ρ - 1)/6 + O(z⁵)

For |z| < 1e-7, first-order approximation sufficient:
  x(z) ≈ z(1 - ρ)
```

### 5. Parameter Calibration Strategy

**Common Approaches:**

1. **Fix β (Most Common)**
   - Equities: β ∈ [0.5, 0.8]
   - FX: β ∈ [0.3, 0.7]
   - Rates: β ∈ [0.0, 0.5]

   Optimize: {α, ρ, ν}

2. **Calibrate All Four**
   - More flexible but less stable
   - Requires good initial guess
   - May overfit to noise

**Initial Guess Heuristics:**
```
Function: sabr_initial_guess(strikes, vols, forward, maturity, fix_β)

  // Find ATM volatility
  atm_idx ← argmin(|strikes - forward|)
  σ_ATM ← vols[atm_idx]

  // Set β
  β ← fix_β if provided else 0.5

  // Initial α: scale ATM vol by forward
  α ← σ_ATM · forward^(1 - β)

  // Estimate ρ from skew
  otm_puts ← strikes where K < 0.95·F
  otm_calls ← strikes where K > 1.05·F

  if both exist:
    skew ← mean(vols[otm_puts]) - mean(vols[otm_calls])
    ρ ← clip(-skew / σ_ATM, -0.7, 0.7)
  else:
    ρ ← -0.3  // Default negative correlation

  // Initial ν: moderate vol-of-vol
  ν ← 0.3

  return SABRParameters(α, β, ρ, ν)
```

---

## Calibration Theory

### 1. Objective Function

**Weighted Least Squares:**
```
min Σᵢ wᵢ · [σ_market,i - σ_model,i(θ)]²
 θ

where:
  θ = model parameters (SVI: {a,b,ρ,m,σ}, SABR: {α,β,ρ,ν})
  wᵢ = weight for quote i
```

**Weight Schemes:**

1. **Uniform Weights:**
   ```
   wᵢ = 1/N
   ```

2. **Vega-Weighted:**
   ```
   wᵢ = vega_i / Σⱼ vega_j
   ```

   Rationale: Higher vega options are more liquid and price-sensitive

3. **Volume-Weighted:**
   ```
   wᵢ = volume_i / Σⱼ volume_j
   ```

   Rationale: More traded strikes are more reliable

**Vega Approximation (for weighting):**
```
vega ≈ F · sqrt(T) · φ(d₁)

where:
  d₁ = [ln(F/K) + 0.5·σ²·T] / (σ·sqrt(T))
  φ(x) = exp(-x²/2) / sqrt(2π)
```

### 2. Regularization

**L2 Regularization (Ridge):**
```
Objective_reg = Σᵢ wᵢ · [σ_market,i - σ_model,i]² + λ · ||θ - θ_prior||²

where:
  λ = regularization strength (typical: 0.001 - 0.01)
  θ_prior = prior parameter values (often initial guess)
```

**Purpose:**
- Prevent overfitting to noisy data
- Stabilize calibration for sparse quotes
- Maintain parameter smoothness across maturities

### 3. Arbitrage Penalty

**Soft Constraint Enforcement:**
```
Objective_arb = LSE + penalty · Σⱼ max(0, -g_j)²

where:
  g_j = arbitrage condition j (should be ≥ 0)
  penalty = large constant (typical: 100 - 10000)
```

**Durrleman Condition Penalty:**
```
For SVI:
  penalty_term = penalty · Σₖ max(0, -g(k))²

where g(k) is evaluated on dense grid k ∈ [-2, 2]
```

### 4. Optimization Algorithms

**L-BFGS-B (Recommended for SVI):**
```
Method: Limited-memory BFGS with bounds
Advantages:
  - Fast convergence for smooth objectives
  - Handles box constraints naturally
  - Low memory footprint

Settings:
  maxiter: 1000
  ftol: 1e-8 (function tolerance)
  gtol: 1e-5 (gradient tolerance)
```

**SLSQP (Sequential Least Squares Programming):**
```
Method: Sequential quadratic programming
Advantages:
  - Handles nonlinear constraints
  - Good for SABR with complex bounds

Settings:
  maxiter: 1000
  ftol: 1e-8
```

**Trust-Region Constrained:**
```
Method: Trust region with constraints
Advantages:
  - Most robust for difficult problems
  - Strict constraint satisfaction

Settings:
  maxiter: 1000
  xtol: 1e-8
```

### 5. Multi-Start Strategy

**Algorithm:**
```
Function: multi_start_calibration(quotes, initial_guess, n_starts)

  best_params ← None
  best_error ← ∞

  For i = 1 to n_starts:
    if i == 1:
      guess ← initial_guess  // Use provided guess
    else:
      guess ← perturb(initial_guess)  // Random perturbation

    try:
      params, error ← optimize(guess, quotes)

      if error < best_error:
        best_params ← params
        best_error ← error

    catch OptimizationError:
      continue  // Try next start

  if best_params is None:
    raise "All optimization runs failed"

  return best_params, best_error
```

**Perturbation Strategy:**
```
For SVI:
  noise ~ Uniform(0.8, 1.2)
  a' ← max(a · noise, ε)
  b' ← max(b · noise, ε)
  ρ' ← clip(ρ · noise, -0.99, 0.99)
  m' ← m · noise
  σ' ← max(σ · noise, ε)

For SABR:
  α' ← max(α · noise, ε)
  β' ← β (if fixed) else clip(β · noise, 0, 1)
  ρ' ← clip(ρ · noise, -0.99, 0.99)
  ν' ← max(ν · noise, ε)
```

### 6. Convergence Diagnostics

**Quality Metrics:**

1. **RMSE (Root Mean Squared Error):**
   ```
   RMSE = sqrt[Σᵢ (σ_market,i - σ_model,i)² / N]

   Target: RMSE < 0.01 (1% or 100bp)
   Good: RMSE < 0.005 (50bp)
   ```

2. **MAE (Mean Absolute Error):**
   ```
   MAE = Σᵢ |σ_market,i - σ_model,i| / N
   ```

3. **Max Error:**
   ```
   Max_Error = max_i |σ_market,i - σ_model,i|

   Target: < 0.02 (200bp)
   ```

4. **R² (Coefficient of Determination):**
   ```
   R² = 1 - SS_res / SS_tot

   where:
     SS_res = Σᵢ (σ_market,i - σ_model,i)²
     SS_tot = Σᵢ (σ_market,i - σ̄_market)²

   Target: R² > 0.95
   ```

5. **Weighted RMSE:**
   ```
   WRMSE = sqrt[Σᵢ wᵢ · (σ_market,i - σ_model,i)²]

   Better reflects fit quality for weighted calibration
   ```

**Convergence Criteria:**
```
Optimization converged if:
  1. Gradient norm < gtol (gradient tolerance)
  AND
  2. Function change < ftol (function tolerance)
  AND
  3. Maximum iterations not exceeded
```

---

## Arbitrage Conditions

### 1. Butterfly Arbitrage

**Condition:** Call prices must be convex in strike.

**Mathematical Statement:**
```
d²C/dK² ≥ 0  for all K

Equivalently (discrete):
  [C(K+h) - C(K)] / h  ≥  [C(K) - C(K-h)] / h
```

**Discrete Approximation:**
```
For strikes K₁ < K₂ < K₃:

d²C/dK² ≈ [C(K₃) - C(K₂)] / (K₃ - K₂) - [C(K₂) - C(K₁)] / (K₂ - K₁)
          ─────────────────────────────────────────────────────────────
                               (K₃ - K₁) / 2

Should be ≥ 0
```

**Relation to Implied Volatility:**
```
For fixed T, butterfly condition equivalent to:

  g(k) = (1 - k·w'(k)/(2w(k)))² - [w'(k)]²/4 · [1/w(k) + 1/4] ≥ 0

where w(k) = σ²(k)·T
```

**Computational Check:**
```
Function: check_butterfly(strikes, call_prices, tolerance)

  Sort strikes and prices in ascending order

  For i = 1 to N-2:
    K_left ← strikes[i-1]
    K_mid ← strikes[i]
    K_right ← strikes[i+1]

    C_left ← call_prices[i-1]
    C_mid ← call_prices[i]
    C_right ← call_prices[i+1]

    h_left ← K_mid - K_left
    h_right ← K_right - K_mid
    h_avg ← (h_left + h_right) / 2

    // Second difference
    d2C ← [(C_right - C_mid) / h_right - (C_mid - C_left) / h_left] / h_avg

    if d2C < -tolerance:
      return False, "Butterfly violation at K=" + K_mid

  return True, "No butterfly arbitrage"
```

### 2. Calendar Arbitrage

**Condition:** Total variance must be non-decreasing in time.

**Mathematical Statement:**
```
For fixed strike K:
  ∂w/∂T ≥ 0

where w(K,T) = σ²_impl(K,T) · T
```

**Discrete Check:**
```
For T₁ < T₂:
  w(K, T₂) ≥ w(K, T₁)  for all K

Equivalently:
  σ²_impl(K, T₂) · T₂  ≥  σ²_impl(K, T₁) · T₁
```

**Algorithm:**
```
Function: check_calendar(strike, maturities, surface, tolerance)

  Sort maturities in ascending order: T₁ < T₂ < ... < Tₙ

  For i = 1 to N-1:
    w_i ← σ²_impl(K, Tᵢ) · Tᵢ
    w_next ← σ²_impl(K, Tᵢ₊₁) · Tᵢ₊₁

    increment ← w_next - w_i

    if increment < -tolerance:
      return False, "Calendar violation between T=" + Tᵢ + " and " + Tᵢ₊₁

  return True, "No calendar arbitrage"
```

**Note on European Options:**
Deep ITM/OTM European options can have ∂C/∂T < 0 due to discounting, but total variance must still be non-decreasing.

### 3. Static Arbitrage Bounds

**Call Price Bounds:**
```
max(0, F - K) ≤ C(K,T) ≤ F

Lower bound: Intrinsic value (discounting omitted for forward)
Upper bound: Cannot exceed forward price
```

**Put-Call Parity:**
```
C(K,T) - P(K,T) = F - K  (for forward contract)

With discounting:
  C(K,T) - P(K,T) = e^(-rT) · (F - K)
```

**Strike Ordering:**
```
For K₁ < K₂:
  C(K₁, T) ≥ C(K₂, T)
  P(K₁, T) ≤ P(K₂, T)

Call prices decrease in strike, put prices increase
```

### 4. Durrleman's Condition (SVI-Specific)

**Necessary and Sufficient for No Butterfly Arbitrage in SVI:**

```
g(k) ≥ 0  for all k ∈ ℝ

where:
  g(k) = (1 - y(k))² - w'(k)² / 4 · (1/w(k) + 1/4)

  y(k) = k · w'(k) / (2w(k))
```

**Equivalent Form:**
```
Define:
  d(k) = w(k) / w'(k)²  (local density)

Then:
  g(k) = (1/w(k) - 1/4) · [w'(k)² - 4w(k)] + (1 - k/(2d(k)))²
```

**Computational Algorithm:**
```
Function: check_durrleman(svi_model, k_grid, tolerance)

  For each k in k_grid:
    w ← svi_model.total_variance(k)
    w_prime ← svi_model.variance_derivative(k)

    // Avoid numerical issues
    if w < ε:
      w ← ε

    // Compute g(k)
    y ← k · w_prime / (2 · w)
    term1 ← (1 - y)²
    term2 ← w_prime² / 4 · (1/w + 0.25)

    g[k] ← term1 - term2

    if g[k] < -tolerance:
      violations.append(k)

  if violations is empty:
    return True, "Durrleman condition satisfied"
  else:
    return False, "Violations at k = " + violations
```

**Grid Selection:**
```
Recommended grid for checking:
  k ∈ [-2, 2] with spacing ≤ 0.01

  Corresponds to strikes:
    K/F ∈ [e^(-2), e^2] ≈ [0.135, 7.389]

  Covers typical option market range
```

### 5. Tolerance Selection

**Numerical Precision:**
```
Strict tolerance:    ε = 1e-8   (for exact enforcement)
Standard tolerance:  ε = 1e-6   (for practical checks)
Loose tolerance:     ε = 1e-4   (for noisy data)
```

**Reasoning:**
- float64 machine epsilon: ~2.22e-16
- Typical optimization convergence: 1e-8
- Market data bid-ask spread: ~1e-4 (0.01% = 1bp)

**Recommended Settings:**
```
Calibration objective: ftol = 1e-8
Parameter bounds: ε = 1e-6
Arbitrage checks: tolerance = 1e-6
Market comparison: threshold = 1e-4 (1bp)
```

---

## Numerical Algorithms

### 1. Implied Volatility to Price Conversion

**Black-Scholes Formula:**
```
C(K,F,T,σ) = F·Φ(d₁) - K·Φ(d₂)

where:
  d₁ = [ln(F/K) + σ²T/2] / (σ·√T)
  d₂ = d₁ - σ·√T
  Φ(·) = standard normal CDF
```

**For Put Options:**
```
P(K,F,T,σ) = K·Φ(-d₂) - F·Φ(-d₁)
```

**Note:** Omits discounting for forward price framework. For spot prices:
```
C = e^(-rT)·[S·e^(rT)·Φ(d₁) - K·Φ(d₂)]
```

### 2. Surface Interpolation

**Time Dimension (Between Maturities):**

**Method: Linear on Total Variance**
```
For T ∈ [T₁, T₂]:
  w(K, T) = w₁ + (T - T₁)/(T₂ - T₁) · (w₂ - w₁)

  where:
    w₁ = σ₁²(K) · T₁
    w₂ = σ₂²(K) · T₂

  Then:
    σ(K, T) = √(w(K,T) / T)
```

**Method: PCHIP (Piecewise Cubic Hermite)**
```
Advantages:
  - Preserves monotonicity
  - C¹ continuous
  - No overshoot

Use scipy.interpolate.PchipInterpolator on total variance
```

**Strike Dimension (Within Maturity):**

Use calibrated model directly (no interpolation needed):
- SVI: σ(K) = √(w(ln(K/F)) / T)
- SABR: σ(K) from Hagan formula

### 3. Surface Extrapolation

**Short Maturity (T < T_min):**

**Strategy 1: Constant Variance**
```
For T < T₁:
  σ(K, T) = σ(K, T₁)  // Use shortest maturity smile
```

**Strategy 2: Linear Decay to ATM**
```
For T < T₁:
  σ(K, T) = σ_ATM + (σ(K, T₁) - σ_ATM) · (T / T₁)

  Rationale: Smile flattens as T → 0
```

**Long Maturity (T > T_max):**

**Strategy 1: Constant Slice**
```
For T > Tₙ:
  σ(K, T) = σ(K, Tₙ)  // Use longest maturity smile
```

**Strategy 2: Flatten to ATM**
```
For T > Tₙ:
  decay = exp(-λ(T - Tₙ))
  σ(K, T) = σ_ATM + (σ(K, Tₙ) - σ_ATM) · decay

  where λ = decay rate (typical: 0.1 - 0.5)
```

### 4. Vega Calculation

**Black-Scholes Vega:**
```
ν = ∂C/∂σ = F·√T·φ(d₁)

where:
  φ(x) = exp(-x²/2) / √(2π)
  d₁ = [ln(F/K) + σ²T/2] / (σ·√T)
```

**Numerical Implementation:**
```python
from scipy.stats import norm

def black_scholes_vega(K, F, T, sigma):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * np.sqrt(T) * norm.pdf(d1)
```

**Properties:**
- Vega is maximum ATM (K = F)
- Vega → 0 as K → 0 or K → ∞
- Vega increases with √T

### 5. Gradient Computation for Calibration

**Analytical Gradients (Faster Convergence):**

For SVI calibration, gradient of σ with respect to parameters:

```
∂σ/∂a = 1/(2σT)

∂σ/∂b = [ρ(k-m) + √((k-m)² + σ²)] / (2σT)

∂σ/∂ρ = b(k-m) / (2σT)

∂σ/∂m = -b[ρ + (k-m)/√((k-m)² + σ²)] / (2σT)

∂σ/∂σ_param = bσ_param / [2σT·√((k-m)² + σ²)]
```

**Numerical Gradients (More Robust):**
```
∂f/∂θᵢ ≈ [f(θ + hₑᵢ) - f(θ - hₑᵢ)] / (2h)

where:
  h = max(|θᵢ|, 1) · √ε_mach  ≈ 1e-8
  eᵢ = unit vector in direction i
```

### 6. Parallel Calibration

**Multi-Core Strategy:**
```python
from concurrent.futures import ProcessPoolExecutor

def calibrate_slice(maturity, quotes):
    engine = CalibrationEngine()
    return engine.calibrate_svi(quotes)

def calibrate_surface_parallel(quotes_by_maturity):
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calibrate_slice, T, quotes): T
            for T, quotes in quotes_by_maturity.items()
        }

        results = {}
        for future in as_completed(futures):
            T = futures[future]
            results[T] = future.result()

    return results
```

**Performance Considerations:**
- Each maturity slice is independent
- Parallelize across N_cpu cores
- Expected speedup: ~N_cpu for N_maturities > N_cpu

---

## References

### Academic Papers

1. **SVI Model:**
   - Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives"
   - Gatheral, J., & Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces"

2. **SABR Model:**
   - Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). "Managing smile risk"
   - West, G. (2005). "Calibration of the SABR model in illiquid markets"

3. **Arbitrage Theory:**
   - Roper, M. (2010). "Arbitrage free implied volatility surfaces"
   - Durrleman, V. (2010). "From implied to spot volatilities"

### Implementation Resources

4. **Numerical Methods:**
   - Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization" (2nd ed.)
   - Press, W. H., et al. (2007). "Numerical Recipes: The Art of Scientific Computing" (3rd ed.)

5. **Volatility Surface:**
   - Lehalle, C.-A., & Rosenbaum, M. (2014). "Complete calibration of a multi-factor local volatility model for FX"
   - Homescu, C. (2011). "Implied volatility surface: Construction methodologies and characteristics"

### Software Libraries

6. **Python Numerical Computing:**
   - NumPy: https://numpy.org/doc/
   - SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
   - Decimal: https://docs.python.org/3/library/decimal.html

7. **Quantitative Finance:**
   - QuantLib: https://www.quantlib.org/
   - py_vollib: https://github.com/vollib/py_vollib

---

## Appendix: Precision Analysis

### Floating-Point Error Propagation

**SVI Total Variance Calculation:**
```
Condition number analysis:
  κ(w) ≈ max(|a|, |b·(ρ·k + √(k² + σ²))|) / |w|

For well-conditioned parameters:
  κ(w) < 10

Expected error:
  |w_computed - w_exact| ≤ κ(w) · ε_mach · |w_exact|
  ≈ 10 · 2.2e-16 · w
  ≈ 2e-15 · w

In volatility units (σ = √(w/T)):
  Δσ ≈ 1e-15 / √T

Conclusion: float64 provides 15 decimal digits of precision,
sufficient for volatility (typically 4-6 significant digits needed)
```

### Decimal vs Float Decision Matrix

| Quantity | Type | Reason |
|----------|------|--------|
| Strike K | Decimal | Exact price (legal contracts) |
| Forward F | Decimal | Exact price (no rounding) |
| Option price | Decimal | Financial settlement |
| Implied vol σ | float64 | Scientific approximation |
| Model parameters | float64 | Optimization efficiency |
| Greeks | Decimal | Risk management precision |

### Numerical Stability Tests

**Test 1: Extreme Strikes**
```
K/F ∈ [0.01, 100] should not cause overflow/underflow
```

**Test 2: Short Maturities**
```
T ∈ [1/365, 30] should maintain 6 significant digits
```

**Test 3: Parameter Extremes**
```
SVI: a ∈ [1e-6, 10], b ∈ [1e-6, 5], etc.
SABR: α ∈ [0.01, 5], ν ∈ [0.01, 3], etc.
```

All tests should pass with relative error < 1e-6.

---

**Document Version:** 1.0
**Last Updated:** 2024
**Authors:** Quantitative Research Team
**Classification:** Technical Specification
