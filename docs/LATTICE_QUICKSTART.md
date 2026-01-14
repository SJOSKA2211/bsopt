# Lattice Models Quick Start Guide

## 🚀 Quick Start (30 seconds)

```python
from src.pricing.lattice import BinomialTreePricer

# Price an American put option
pricer = BinomialTreePricer(
    spot=100,           # Current stock price
    strike=100,         # Strike price
    maturity=1.0,       # 1 year to expiration
    volatility=0.25,    # 25% volatility
    rate=0.05,          # 5% risk-free rate
    dividend=0.02,      # 2% dividend yield
    option_type='put',
    exercise_type='american',
    n_steps=200         # Tree depth
)

# Get price and Greeks
price = pricer.price()
greeks = pricer.get_greeks()

print(f"Option Price: ${price:.4f}")
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
```

## 📊 What's Included

### Models
- **Binomial Tree (CRR)**: Industry-standard, fast, reliable
- **Trinomial Tree**: More stable for high volatility

### Features
- ✅ American & European exercise
- ✅ Call & Put options
- ✅ All Greeks (Delta, Gamma, Vega, Theta, Rho)
- ✅ Early exercise boundary detection
- ✅ Convergence validation
- ✅ Vectorized (50-70% faster)

## 📈 Common Use Cases

### 1. Price American Put

```python
from src.pricing.lattice import BinomialTreePricer

pricer = BinomialTreePricer(
    spot=100, strike=110, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02,
    option_type='put',
    exercise_type='american',
    n_steps=300
)

price = pricer.price()
boundary = pricer.get_early_exercise_boundary()

print(f"American Put: ${price:.4f}")
print(f"Exercise below: ${boundary[0]:.2f}")
```

### 2. Calculate Greeks

```python
pricer = BinomialTreePricer(
    spot=100, strike=100, maturity=0.25,
    volatility=0.25, rate=0.05,
    option_type='call',
    n_steps=200
)

greeks = pricer.get_greeks()

print(f"Delta:  {greeks.delta:.4f}  (hedge ratio)")
print(f"Gamma:  {greeks.gamma:.4f}  (convexity)")
print(f"Vega:   {greeks.vega:.4f}   (vol sensitivity)")
print(f"Theta:  {greeks.theta:.4f}  (time decay/day)")
print(f"Rho:    {greeks.rho:.4f}    (rate sensitivity)")
```

### 3. Validate Convergence

```python
from src.pricing.lattice import validate_convergence

results = validate_convergence(
    spot=100, strike=100, maturity=1.0,
    volatility=0.20, rate=0.05,
    option_type='call',
    step_sizes=[50, 100, 200, 500]
)

# Compare to Black-Scholes
bs_price = results['black_scholes']
for i, n in enumerate(results['step_sizes']):
    price = results['binomial'][i]
    error = results['binomial_errors'][i]
    print(f"N={n:3d}: ${price:.6f} (error: {error/bs_price*100:.3f}%)")
```

### 4. Compare Binomial vs Trinomial

```python
from src.pricing.lattice import BinomialTreePricer, TrinomialTreePricer
import time

# Binomial
start = time.perf_counter()
bin_price = BinomialTreePricer(
    spot=100, strike=100, maturity=1.0,
    volatility=0.30, rate=0.05,
    option_type='put', exercise_type='american',
    n_steps=300
).price()
bin_time = (time.perf_counter() - start) * 1000

# Trinomial
start = time.perf_counter()
tri_price = TrinomialTreePricer(
    spot=100, strike=100, maturity=1.0,
    volatility=0.30, rate=0.05,
    option_type='put', exercise_type='american',
    n_steps=300
).price()
tri_time = (time.perf_counter() - start) * 1000

print(f"Binomial:  ${bin_price:.4f} in {bin_time:.1f}ms")
print(f"Trinomial: ${tri_price:.4f} in {tri_time:.1f}ms")
print(f"Difference: ${abs(bin_price - tri_price):.4f}")
```

## ⚡ Performance Guide

### Recommended Settings

| Use Case | Steps | Time | Accuracy |
|----------|-------|------|----------|
| Quick estimate | 50 | ~1ms | 0.5% |
| Standard pricing | 200 | ~8ms | 0.1% |
| High accuracy | 500 | ~50ms | 0.04% |
| Ultra precision | 1000 | ~200ms | 0.02% |

### When to Use What

**Use Binomial when:**
- Standard market conditions (σ < 50%)
- Speed is important
- Industry standard needed

**Use Trinomial when:**
- High volatility (σ > 50%)
- Pricing barrier options
- Need smoother convergence

## 🔧 Advanced Features

### Early Exercise Boundary

```python
pricer = BinomialTreePricer(
    spot=100, strike=110, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.08,
    option_type='put', exercise_type='american',
    n_steps=300
)

boundary = pricer.get_early_exercise_boundary()

# boundary[i] = critical spot at time step i
# Exercise if S < boundary[i]
import numpy as np
time_points = np.linspace(0, 1.0, len(boundary))

for i in [0, len(boundary)//2, len(boundary)-1]:
    t = time_points[i]
    S_critical = boundary[i]
    print(f"t={t:.2f}: Exercise if S < ${S_critical:.2f}")
```

### Memory-Efficient Mode

```python
# For very deep trees (N > 1000)
pricer = BinomialTreePricer(
    spot=100, strike=100, maturity=1.0,
    volatility=0.20, rate=0.05,
    option_type='call',
    n_steps=2000
)

# Use memory-efficient pricing
price = pricer._price_memory_efficient()
# Note: Cannot calculate Greeks from this
```

### Custom Validation

```python
from src.pricing.black_scholes import BSParameters, BlackScholesEngine

# Price with lattice
lattice_price = BinomialTreePricer(
    spot=100, strike=100, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02,
    option_type='call', exercise_type='european',
    n_steps=500
).price()

# Price with Black-Scholes
bs_params = BSParameters(
    spot=100, strike=100, maturity=1.0,
    volatility=0.20, rate=0.05, dividend=0.02
)
bs_price = BlackScholesEngine.price_call(bs_params)

# Compare
error = abs(lattice_price - bs_price)
print(f"Lattice:       ${lattice_price:.6f}")
print(f"Black-Scholes: ${bs_price:.6f}")
print(f"Error:         ${error:.6f} ({error/bs_price*100:.3f}%)")
```

## 🧪 Testing

```bash
# Run all lattice tests
pytest tests/test_lattice.py -v

# Run specific test category
pytest tests/test_lattice.py::TestBinomialTreePricer -v

# Run performance benchmarks
pytest tests/test_lattice.py::TestPerformance -v

# Check coverage
pytest tests/test_lattice.py --cov=src.pricing.lattice
```

## 📚 Examples

Run comprehensive examples:

```bash
python examples/lattice_examples.py
```

This will run 6 examples covering:
1. Basic pricing
2. American options with early exercise
3. Convergence studies
4. Trinomial comparison
5. Performance benchmarks
6. Greeks analysis

## 🎯 Common Pitfalls

### 1. Too Few Steps

```python
# ❌ BAD: Too few steps for accurate Greeks
pricer = BinomialTreePricer(..., n_steps=20)
greeks = pricer.get_greeks()  # Noisy results!

# ✅ GOOD: Use at least 100 steps for Greeks
pricer = BinomialTreePricer(..., n_steps=100)
greeks = pricer.get_greeks()  # Smooth results
```

### 2. Very Low Volatility

```python
# ❌ BAD: May violate no-arbitrage
pricer = BinomialTreePricer(
    ..., volatility=0.001, n_steps=50
)

# ✅ GOOD: Increase steps or use higher volatility
pricer = BinomialTreePricer(
    ..., volatility=0.001, n_steps=500
)
```

### 3. Forgetting to Check Exercise Type

```python
# ❌ BAD: American call with no dividend
# (will be same as European, wasted computation)
pricer = BinomialTreePricer(
    ..., dividend=0.0,
    option_type='call',
    exercise_type='american'
)

# ✅ GOOD: Use European for calls with q=0
pricer = BinomialTreePricer(
    ..., dividend=0.0,
    option_type='call',
    exercise_type='european'
)
```

## 📖 Further Reading

- **Full Documentation**: `docs/LATTICE_MODELS.md`
- **API Reference**: See docstrings in `src/pricing/lattice.py`
- **Implementation Summary**: `LATTICE_IMPLEMENTATION_SUMMARY.md`

## 💡 Tips

1. **Start with 200 steps** for most applications
2. **Use European exercise** for calls without dividends
3. **Check convergence** with increasing steps
4. **Validate against Black-Scholes** for European options
5. **Profile performance** for production use cases

## 🎓 Theory

### Binomial Tree (CRR)

```
Tree Parameters:
  u = exp(σ√Δt)              Up factor
  d = 1/u                     Down factor
  p = (e^((r-q)Δt) - d)/(u-d) Risk-neutral probability

Backward Induction:
  V[i,j] = e^(-rΔt) * [p*V[i+1,j+1] + (1-p)*V[i+1,j]]

American:
  V[i,j] = max(continuation, intrinsic_value)
```

### Convergence

```
Error = O(1/N) for binomial trees
At N=500: ~0.04% error vs Black-Scholes
```

## 🆘 Support

If you encounter issues:

1. Check parameter validity (spot > 0, strike > 0, etc.)
2. Verify volatility isn't too low/high
3. Increase n_steps if results seem unstable
4. Review docstrings for parameter meanings
5. Check examples for similar use cases

---

**Ready to price options? Start with the Quick Start example above!** 🚀
