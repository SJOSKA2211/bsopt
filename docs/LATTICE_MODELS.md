# Lattice-Based Option Pricing Models

## Overview

This module implements production-ready lattice-based option pricing models for both European and American-style options. The implementation includes:

- **Cox-Ross-Rubinstein (CRR) Binomial Tree**: Industry-standard binomial tree model
- **Standard Trinomial Tree**: Extended model with three branches per node
- **Vectorized Implementation**: Optimized performance using NumPy
- **Greeks Calculation**: All standard sensitivities via finite differences
- **Early Exercise Detection**: Boundary identification for American options
- **Convergence Validation**: Tests against Black-Scholes analytical solutions

## Features

### 1. Binomial Tree Pricer (CRR)

The Cox-Ross-Rubinstein model constructs a recombining binomial tree where asset prices evolve in discrete time steps.

**Key Characteristics:**
- Recombining tree structure (efficient memory usage)
- Risk-neutral probability calibrated to match volatility
- O(N²) time complexity, O(N²) space complexity
- Converges to Black-Scholes as N → ∞

**Parameterization:**
```
dt = T / N                          # Time step
u = exp(σ√dt)                       # Up factor
d = 1/u = exp(-σ√dt)               # Down factor
p = (exp((r-q)dt) - d) / (u - d)   # Risk-neutral probability
```

**Usage:**
```python
from src.pricing.lattice import BinomialTreePricer

pricer = BinomialTreePricer(
    spot=100.0,           # Current stock price
    strike=100.0,         # Strike price
    maturity=1.0,         # Time to maturity (years)
    volatility=0.25,      # Annualized volatility
    rate=0.05,            # Risk-free rate
    dividend=0.02,        # Dividend yield
    option_type='call',   # 'call' or 'put'
    exercise_type='american',  # 'european' or 'american'
    n_steps=200           # Number of time steps
)

price = pricer.price()
greeks = pricer.get_greeks()
boundary = pricer.get_early_exercise_boundary()
```

### 2. Trinomial Tree Pricer

The trinomial model extends binomial trees by adding a middle branch at each node, providing additional flexibility.

**Key Characteristics:**
- Three branches per node (up, middle, down)
- More stable for high volatility scenarios
- Slightly slower than binomial (1.3-1.5x)
- Often better convergence properties

**Parameterization:**
```
dt = T / N
dx = σ√(3dt)                    # Log-space spacing
u = exp(dx)                      # Up factor
d = exp(-dx)                     # Down factor
m = 1                            # Middle factor (no change)

# Risk-neutral probabilities (match mean and variance)
ν = r - q - 0.5σ²
p_u = 0.5(σ²dt + ν²dt²)/dx² + 0.5νdt/dx
p_d = 0.5(σ²dt + ν²dt²)/dx² - 0.5νdt/dx
p_m = 1 - p_u - p_d
```

**Usage:**
```python
from src.pricing.lattice import TrinomialTreePricer

pricer = TrinomialTreePricer(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    volatility=0.25,
    rate=0.05,
    dividend=0.02,
    option_type='put',
    exercise_type='american',
    n_steps=200
)

price = pricer.price()
greeks = pricer.get_greeks()
```

## Backward Induction Algorithm

Both models use backward induction to value options:

### European Options

```python
# Terminal payoffs at maturity
V[N, j] = max(S[N, j] - K, 0)  # Call
V[N, j] = max(K - S[N, j], 0)  # Put

# Backward induction
for i in range(N-1, -1, -1):
    for j in range(states[i]):
        # Continuation value (discounted expected value)
        V[i, j] = exp(-r·dt) * E[V[i+1]]
```

### American Options

```python
# Terminal payoffs
V[N, j] = max(S[N, j] - K, 0)  # Call
V[N, j] = max(K - S[N, j], 0)  # Put

# Backward induction with early exercise
for i in range(N-1, -1, -1):
    for j in range(states[i]):
        # Continuation value
        continuation = exp(-r·dt) * E[V[i+1]]

        # Intrinsic value (immediate exercise)
        intrinsic = max(S[i, j] - K, 0)  # Call
        intrinsic = max(K - S[i, j], 0)  # Put

        # Take maximum
        V[i, j] = max(continuation, intrinsic)
```

## Greeks Calculation

Greeks are computed using finite difference approximations from the tree structure:

### Delta (Δ)
Rate of change of option price with respect to spot price.

```python
Δ = (V[1, 1] - V[1, 0]) / (S[1, 1] - S[1, 0])
```

**Interpretation:**
- Call: 0 ≤ Δ ≤ 1 (hedge ratio)
- Put: -1 ≤ Δ ≤ 0
- ATM options: Δ ≈ 0.5 (call), Δ ≈ -0.5 (put)

### Gamma (Γ)
Rate of change of delta with respect to spot price.

```python
Δ_up = (V[2, 2] - V[2, 1]) / (S[2, 2] - S[2, 1])
Δ_down = (V[2, 1] - V[2, 0]) / (S[2, 1] - S[2, 0])
Γ = (Δ_up - Δ_down) / ((S[2, 2] - S[2, 0]) / 2)
```

**Interpretation:**
- Always positive for long options
- Highest for ATM options
- Measures convexity of option price

### Vega (ν)
Sensitivity to volatility changes ($ per 1% volatility change).

```python
# Recompute with σ + 0.01
V_up = price(σ + 0.01)
V_base = price(σ)
ν = V_up - V_base
```

**Interpretation:**
- Always positive for long options
- Highest for ATM options
- Represents exposure to implied volatility changes

### Theta (Θ)
Time decay ($ per day).

```python
Θ = (V[2, 1] - V[0, 0]) / (2·dt) / 365
```

**Interpretation:**
- Negative for long options (value decays over time)
- Largest magnitude for ATM options near expiration
- Represents time value erosion

### Rho (ρ)
Sensitivity to interest rate changes ($ per 1% rate change).

```python
# Recompute with r + 0.01
V_up = price(r + 0.01)
V_base = price(r)
ρ = V_up - V_base
```

**Interpretation:**
- Call: ρ > 0 (benefits from rate increases)
- Put: ρ < 0 (hurt by rate increases)
- More significant for longer maturities

## Early Exercise Boundary

For American options, the early exercise boundary is the critical spot price at each time step where immediate exercise becomes optimal.

```python
boundary = pricer.get_early_exercise_boundary()

# Returns array of shape (n_steps+1,)
# boundary[i] = critical spot price at time step i
```

**Characteristics:**
- **American Put**: Boundary decreases over time (generally)
- **American Call** (with dividends): Boundary increases near ex-dividend dates
- **American Call** (no dividends): Should not exercise early (boundary at infinity)

**Example:**
```python
pricer = BinomialTreePricer(
    spot=100, strike=110, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.08,
    option_type='put', exercise_type='american',
    n_steps=300
)

boundary = pricer.get_early_exercise_boundary()

# At time t, if S < boundary[t], exercise immediately
```

## Convergence and Validation

### Convergence to Black-Scholes

For European options, lattice prices converge to Black-Scholes as the number of steps increases:

```python
from src.pricing.lattice import validate_convergence

results = validate_convergence(
    spot=100, strike=100, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02,
    option_type='call',
    step_sizes=[50, 100, 200, 500, 1000]
)

# Results contain:
# - Black-Scholes analytical price
# - Binomial prices for each step size
# - Trinomial prices for each step size
# - Errors for both methods
```

**Expected Convergence:**
- Error decreases as O(1/N) for binomial trees
- Error decreases as O(1/N²) for trinomial trees (theoretical)
- At N=500, typical error < 0.1% for standard parameters

### Validation Tests

The implementation includes comprehensive tests:

```python
# Run all tests
pytest tests/test_lattice.py -v

# Test categories:
# 1. Parameter validation
# 2. Tree construction (recombination, probabilities)
# 3. European option pricing (vs Black-Scholes)
# 4. American option pricing (premium validation)
# 5. Greeks calculation (range checks, sign checks)
# 6. Convergence analysis
# 7. Performance benchmarks
# 8. Edge cases (deep ITM/OTM, high/low volatility)
```

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Tree Construction | O(N²) | O(N²) |
| Pricing | O(N²) | O(N²) |
| Greeks (all) | O(N²) | O(N²) |
| Memory-Efficient | O(N²) | O(N) |

### Benchmarks

Typical performance on modern hardware:

| Steps | Binomial Time | Trinomial Time | Nodes |
|-------|--------------|----------------|-------|
| 50 | ~1 ms | ~2 ms | 1,326 |
| 100 | ~3 ms | ~5 ms | 5,151 |
| 200 | ~8 ms | ~12 ms | 20,301 |
| 500 | ~50 ms | ~70 ms | 125,751 |
| 1000 | ~200 ms | ~280 ms | 501,501 |

**Optimization Tips:**
- Use binomial for standard cases (faster)
- Use trinomial for high volatility or barrier options
- For N > 1000, use memory-efficient mode
- Vectorized implementation is 50-70% faster than loops

## American Option Pricing

### When to Exercise Early

**American Put:**
- Exercise when spot falls significantly below strike
- More valuable with high interest rates (carrying cost of strike)
- More valuable with low/no dividends

**American Call:**
- With dividends: May exercise just before ex-dividend date
- Without dividends: Never optimal to exercise early
- Premium ≈ 0 for calls with q = 0

### Example: American Put Premium

```python
# European put
euro_put = BinomialTreePricer(
    spot=100, strike=110, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02,
    option_type='put', exercise_type='european',
    n_steps=200
)
euro_price = euro_put.price()

# American put
amer_put = BinomialTreePricer(
    spot=100, strike=110, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02,
    option_type='put', exercise_type='american',
    n_steps=200
)
amer_price = amer_put.price()

premium = amer_price - euro_price
print(f"Early exercise premium: ${premium:.4f}")
# Output: Early exercise premium: $0.3356
```

## Choosing Between Binomial and Trinomial

### Use Binomial When:
- Standard market conditions (volatility < 50%)
- Speed is critical
- Industry-standard methodology required
- Pricing vanilla American options

### Use Trinomial When:
- High volatility scenarios (σ > 50%)
- Pricing barrier options
- Need smoother convergence
- Three-state Markov model is natural

### Performance vs Accuracy Trade-off

```
Binomial (200 steps):
  - Time: ~8 ms
  - Error: ~0.02%
  - Recommended for production

Trinomial (150 steps):
  - Time: ~8 ms
  - Error: ~0.02%
  - Equivalent accuracy, fewer steps needed

Binomial (500 steps):
  - Time: ~50 ms
  - Error: ~0.005%
  - High accuracy for critical valuations
```

## Common Pitfalls and Solutions

### 1. Negative Probabilities

**Problem:** Very low volatility or large time steps can cause p < 0 or p > 1.

**Solution:**
```python
# Check volatility term
vol_term = sigma * sqrt(dt)
if vol_term > 0.5:
    # Increase steps or use trinomial
    n_steps = int(T * sigma**2 * 100)
```

### 2. Slow Convergence

**Problem:** Oscillating prices as N increases (especially for digital options).

**Solution:**
- Use averaging: `(price[N] + price[N+1]) / 2`
- Use Richardson extrapolation
- Increase steps significantly

### 3. Greeks Noise

**Problem:** Greeks are noisy for small N.

**Solution:**
- Use N ≥ 100 for Greeks calculation
- Consider analytical Greeks for European options
- Smooth using multiple evaluations

### 4. Memory Issues

**Problem:** Tree requires too much memory for large N.

**Solution:**
```python
# Use memory-efficient mode
price = pricer._price_memory_efficient()
# Note: Cannot compute Greeks from this
```

## References

1. **Cox, J. C., Ross, S. A., & Rubinstein, M. (1979)**
   "Option pricing: A simplified approach"
   *Journal of Financial Economics*, 7(3), 229-263.

2. **Jarrow, R., & Rudd, A. (1983)**
   "Option Pricing"
   Homewood, IL: Richard D. Irwin.

3. **Hull, J. C. (2018)**
   "Options, Futures, and Other Derivatives" (10th ed.)
   Pearson Education.

4. **Wilmott, P. (2006)**
   "Paul Wilmott on Quantitative Finance" (2nd ed.)
   John Wiley & Sons.

## API Reference

### BinomialTreePricer

```python
class BinomialTreePricer:
    def __init__(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
        option_type: Literal['call', 'put'] = 'call',
        exercise_type: Literal['european', 'american'] = 'european',
        n_steps: int = 100
    )

    def price() -> float:
        """Calculate option price via backward induction."""

    def build_tree() -> np.ndarray:
        """Build and return forward price lattice."""

    def get_greeks() -> OptionGreeks:
        """Calculate all Greeks via finite differences."""

    def get_early_exercise_boundary() -> Optional[np.ndarray]:
        """Calculate early exercise boundary (American only)."""
```

### TrinomialTreePricer

```python
class TrinomialTreePricer:
    def __init__(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        rate: float,
        dividend: float = 0.0,
        option_type: Literal['call', 'put'] = 'call',
        exercise_type: Literal['european', 'american'] = 'european',
        n_steps: int = 100
    )

    def price() -> float:
        """Calculate option price via backward induction."""

    def build_tree() -> np.ndarray:
        """Build and return forward price lattice."""

    def get_greeks() -> OptionGreeks:
        """Calculate all Greeks via finite differences."""
```

### Utility Functions

```python
def validate_convergence(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float = 0.0,
    option_type: Literal['call', 'put'] = 'call',
    step_sizes: list = None
) -> dict:
    """Test convergence to Black-Scholes for varying step sizes."""
```

## Examples

See `examples/lattice_examples.py` for comprehensive examples covering:
- Basic option pricing
- American options with early exercise
- Convergence studies
- Greeks analysis
- Performance benchmarks
- Comparison between binomial and trinomial methods

Run examples:
```bash
python examples/lattice_examples.py
```
