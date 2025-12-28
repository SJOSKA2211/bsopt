# Crank-Nicolson Solver - Quick Reference

## Import and Basic Usage

```python
from src.pricing.finite_difference import CrankNicolsonSolver

# Create solver instance
solver = CrankNicolsonSolver(
    spot=100.0,         # Current asset price
    strike=100.0,       # Strike price
    maturity=1.0,       # Time to maturity (years)
    volatility=0.2,     # Annualized volatility (20%)
    rate=0.05,          # Risk-free rate (5%)
    dividend=0.02,      # Dividend yield (2%)
    option_type='call', # 'call' or 'put'
    n_spots=100,        # Spatial grid points
    n_time=100          # Time steps
)

# Solve for option price
price = solver.solve()
print(f"Option Price: ${price:.4f}")

# Calculate Greeks
greeks = solver.get_greeks()
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega: {greeks.vega:.4f}")
print(f"Theta: {greeks.theta:.4f}")
print(f"Rho: {greeks.rho:.4f}")
```

## The Black-Scholes PDE

```
∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0
```

## Crank-Nicolson Scheme

**Matrix form:**
```
A * V^(n-1) = B * V^n
```

**Coefficients at grid point i:**
```
α_i = 0.25 * dt * (σ²*S_i²/dS² - (r-q)*S_i/(2*dS))
β_i = -0.5 * dt * (σ²*S_i²/dS² + r)
γ_i = 0.25 * dt * (σ²*S_i²/dS² + (r-q)*S_i/(2*dS))
```

**Matrix A (implicit):**
```
Lower diagonal: -α
Main diagonal:  1 - β
Upper diagonal: -γ
```

**Matrix B (explicit):**
```
Lower diagonal: α
Main diagonal:  1 + β
Upper diagonal: γ
```

## Boundary Conditions

**Terminal (t=T):**
- Call: `V(S,T) = max(S - K, 0)`
- Put: `V(S,T) = max(K - S, 0)`

**Lower (S=0):**
- Call: `V(0,t) = 0`
- Put: `V(0,t) = K*exp(-r*τ)`

**Upper (S=S_max):**
- Call: `V(S_max,t) = S_max - K*exp(-r*τ)`
- Put: `V(S_max,t) = 0`

## Key Properties

**Stability:** Unconditionally stable
- No CFL condition required
- Stable for any dt and dS

**Accuracy:**
- Spatial: O(dS²)
- Temporal: O(dt²)
- Overall: Second-order in both dimensions

**Computational Cost:**
- Time complexity: O(M*N) where M=n_spots, N=n_time
- Space complexity: O(M)
- Single solve: ~1-10ms for typical grids

## Grid Selection Guidelines

**Spatial points (n_spots):**
- Minimum: 50 points
- Standard: 100-200 points
- High precision: 500+ points

**Time steps (n_time):**
- Rule of thumb: n_time ≥ n_spots
- For high accuracy: n_time = 2*n_spots to 5*n_spots

**Domain:**
- S_min = 0.01 (avoid singularity at 0)
- S_max = 3*K (adequate far-field coverage)

## Greeks Formulas

**Delta (∂V/∂S):**
```python
Δ = (V(S+h) - V(S-h)) / (2h)
```

**Gamma (∂²V/∂S²):**
```python
Γ = (V(S+h) - 2V(S) + V(S-h)) / h²
```

**Vega (∂V/∂σ):**
```python
ν = (V(σ+0.01) - V(σ-0.01)) / 0.02
```

**Theta (∂V/∂t):**
```python
Θ = (V(T-1/365) - V(T)) / (1/365)
```

**Rho (∂V/∂r):**
```python
ρ = (V(r+0.0001) - V(r-0.0001)) / 0.0002
```

## Example Use Cases

### 1. European Call Option
```python
solver = CrankNicolsonSolver(
    spot=100, strike=100, maturity=1.0,
    volatility=0.2, rate=0.05, dividend=0.0,
    option_type='call', n_spots=200, n_time=500
)
price = solver.solve()
```

### 2. European Put Option
```python
solver = CrankNicolsonSolver(
    spot=100, strike=100, maturity=1.0,
    volatility=0.2, rate=0.05, dividend=0.0,
    option_type='put', n_spots=200, n_time=500
)
price = solver.solve()
```

### 3. Dividend-Paying Stock
```python
solver = CrankNicolsonSolver(
    spot=100, strike=100, maturity=1.0,
    volatility=0.2, rate=0.05, dividend=0.03,  # 3% dividend
    option_type='call', n_spots=200, n_time=500
)
price = solver.solve()
```

### 4. Short-dated Option
```python
solver = CrankNicolsonSolver(
    spot=100, strike=100, maturity=0.083,  # 1 month
    volatility=0.2, rate=0.05, dividend=0.0,
    option_type='call', n_spots=150, n_time=300
)
price = solver.solve()
```

### 5. High Volatility
```python
solver = CrankNicolsonSolver(
    spot=100, strike=100, maturity=1.0,
    volatility=0.5,  # 50% volatility
    rate=0.05, dividend=0.0,
    option_type='call', n_spots=250, n_time=500
)
price = solver.solve()
```

## Diagnostics

```python
diag = solver.get_diagnostics()

print(f"Scheme: {diag['scheme']}")
print(f"Grid: {diag['grid_spacing']['n_spots']}x{diag['grid_spacing']['n_time']}")
print(f"dS: {diag['grid_spacing']['dS']:.6f}")
print(f"dt: {diag['grid_spacing']['dt']:.6f}")
print(f"Mesh ratio: {diag['stability']['mesh_ratio']:.6f}")
print(f"Stable: {diag['stability']['is_stable']}")
```

## Validation Checklist

1. **Zero maturity:** Price equals intrinsic value
2. **Put-Call Parity:** C - P = S*exp(-q*T) - K*exp(-r*T)
3. **Monotonicity:** Call price increases with S
4. **Convergence:** Price stabilizes as grid refines
5. **Boundary values:** Correct limits at S=0 and S=∞

## Performance Tips

1. **Start with coarse grid:** Use 100x100 for initial testing
2. **Refine as needed:** Increase to 200x500 for production
3. **Cache solver instances:** Reuse for similar parameters
4. **Greeks are expensive:** Each requires 2-3 additional solves
5. **Monitor convergence:** Check that price stabilizes with finer grids

## Common Pitfalls

1. **Too coarse grid:** May miss important features near ATM
2. **Too fine grid:** Diminishing returns, increased computation time
3. **Small S_max:** Underestimates deep ITM call values
4. **Large dt:** Reduces temporal accuracy despite stability
5. **Ignoring dividends:** Can lead to significant pricing errors

## Error Estimation

**Expected error:**
```
Error ≈ C₁*dt² + C₂*dS²
```

**Rough estimate:**
```
For 100x100 grid with σ=0.2, T=1.0, K=100:
  dS ≈ 3.0
  dt ≈ 0.01
  Expected error: ~0.1% to 1% of option value
```

**Richardson extrapolation:**
```python
# Solve with two grid sizes
V_coarse = solver_100.solve()
V_fine = solver_200.solve()

# Estimate error (assuming quadratic convergence)
error_estimate = abs(V_fine - V_coarse) / 3
```

## File Locations

- **Implementation:** `/home/kamau/comparison/src/pricing/finite_difference.py`
- **Documentation:** `/home/kamau/comparison/CRANK_NICOLSON_DOCUMENTATION.md`
- **Test script:** `/home/kamau/comparison/test_fdm_minimal.py`
