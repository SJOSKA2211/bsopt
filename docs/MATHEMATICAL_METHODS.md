# Mathematical Methods for Option Pricing

## Table of Contents

1. [Introduction](#introduction)
2. [Black-Scholes-Merton Model](#black-scholes-merton-model)
3. [Greeks Derivations](#greeks-derivations)
4. [Finite Difference Methods](#finite-difference-methods)
5. [Monte Carlo Simulation](#monte-carlo-simulation)
6. [Lattice Models](#lattice-models)
7. [Implied Volatility](#implied-volatility)
8. [Volatility Surface Modeling](#volatility-surface-modeling)
9. [Numerical Accuracy and Stability](#numerical-accuracy-and-stability)
10. [References](#references)

## Introduction

This document provides a comprehensive mathematical foundation for the option pricing methods implemented in the Black-Scholes Advanced Option Pricing Platform. Each section includes theoretical derivations, numerical implementation details, and accuracy considerations.

### Mathematical Notation

| Symbol | Description |
|--------|-------------|
| S | Current spot price of the underlying asset |
| K | Strike price of the option |
| T | Time to maturity (in years) |
| t | Current time |
| τ = T - t | Time remaining to maturity |
| σ | Volatility (annualized standard deviation of returns) |
| r | Risk-free interest rate (continuously compounded) |
| q | Continuous dividend yield |
| C(S,t) | Call option value |
| P(S,t) | Put option value |
| N(x) | Cumulative standard normal distribution function |
| n(x) | Standard normal probability density function |

## Black-Scholes-Merton Model

### Assumptions

The Black-Scholes-Merton (BSM) model rests on the following assumptions:

1. **Geometric Brownian Motion**: The underlying asset price follows:
   $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
   where $W_t$ is a standard Brownian motion.

2. **No Arbitrage**: Markets are frictionless and complete.

3. **Constant Parameters**: Volatility σ and risk-free rate r are constant.

4. **Continuous Trading**: Assets can be traded continuously without transaction costs.

5. **No Dividends** (in the basic model; extended to include continuous dividend yield q).

6. **Log-normal Distribution**: Asset prices at maturity are log-normally distributed:
   $$S_T \sim \text{LogNormal}\left(\ln S_0 + (r - q - \frac{\sigma^2}{2})T, \sigma^2 T\right)$$

### Black-Scholes PDE

Under the risk-neutral measure, the option value V(S,t) satisfies the Black-Scholes partial differential equation:

$$\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$

**Boundary Conditions** (European Call):
- Terminal condition: $C(S,T) = \max(S_T - K, 0)$
- Lower boundary: $C(0,t) = 0$
- Upper boundary: $C(S,t) \to S e^{-q(T-t)} - K e^{-r(T-t)}$ as $S \to \infty$

### Closed-Form Solution

#### d₁ and d₂ Parameters

$$d_1 = \frac{\ln(S/K) + (r - q + \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}$$

$$d_2 = d_1 - \sigma\sqrt{\tau} = \frac{\ln(S/K) + (r - q - \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}$$

**Interpretation**:
- $d_1$: Represents the sensitivity of the option price to the underlying (related to delta)
- $d_2$: Represents the risk-adjusted probability that the option finishes in-the-money
- Both are dimensionless and represent standardized distances in the log-normal distribution

#### European Call Option

$$C(S,t) = S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)$$

**Derivation**:
The call price is the discounted expected payoff under the risk-neutral measure:
$$C = e^{-r\tau} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]$$

Using the log-normal distribution of $S_T$:
$$C = e^{-r\tau} \int_K^{\infty} (S_T - K) \phi(S_T) dS_T$$

where $\phi(S_T)$ is the risk-neutral probability density. After integration (change of variables):
$$C = S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)$$

#### European Put Option

$$P(S,t) = K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1)$$

**Derivation via Put-Call Parity**:
$$C - P = S e^{-q\tau} - K e^{-r\tau}$$

Therefore:
$$P = C - S e^{-q\tau} + K e^{-r\tau}$$

Substituting the call formula and simplifying:
$$P = K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1)$$

### Put-Call Parity

**Theorem**: For European options with the same strike and maturity:
$$C - P = S e^{-q\tau} - K e^{-r\tau}$$

**Proof**:
Consider two portfolios:
- **Portfolio A**: Long call + $K e^{-r\tau}$ in cash
- **Portfolio B**: Long put + $S e^{-q\tau}$ worth of stock (dividend-adjusted)

At maturity T:
- If $S_T > K$: Portfolio A = $(S_T - K) + K = S_T$, Portfolio B = $0 + S_T = S_T$
- If $S_T \leq K$: Portfolio A = $0 + K = K$, Portfolio B = $(K - S_T) + S_T = K$

Both portfolios have identical payoffs → they must have the same value today.

**Verification in Code**:
```python
def verify_put_call_parity(params: BSParameters, tolerance: float = 1e-10) -> bool:
    call_price = BlackScholesEngine.price_call(params)
    put_price = BlackScholesEngine.price_put(params)

    left_side = call_price - put_price
    right_side = params.spot * np.exp(-params.dividend * params.maturity) - \
                 params.strike * np.exp(-params.rate * params.maturity)

    return abs(left_side - right_side) < tolerance
```

## Greeks Derivations

The Greeks measure the sensitivity of option prices to various parameters. All Greeks are partial derivatives of the option value.

### Delta (Δ)

**Definition**: Rate of change of option price with respect to underlying price.
$$\Delta = \frac{\partial V}{\partial S}$$

**Call Delta**:
$$\Delta_C = e^{-q\tau} N(d_1)$$

**Derivation**:
$$\frac{\partial C}{\partial S} = \frac{\partial}{\partial S}\left[S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)\right]$$

Using the chain rule and $\frac{\partial d_1}{\partial S} = \frac{1}{S\sigma\sqrt{\tau}}$:
$$= e^{-q\tau} N(d_1) + S e^{-q\tau} n(d_1) \frac{1}{S\sigma\sqrt{\tau}} - K e^{-r\tau} n(d_2) \frac{1}{S\sigma\sqrt{\tau}}$$

Since $S e^{-q\tau} n(d_1) = K e^{-r\tau} n(d_2)$ (spot-strike duality), the second and third terms cancel:
$$\Delta_C = e^{-q\tau} N(d_1)$$

**Put Delta**:
$$\Delta_P = -e^{-q\tau} N(-d_1) = e^{-q\tau} (N(d_1) - 1)$$

**Range**:
- Call: $0 \leq \Delta_C \leq 1$ (approaches 1 for deep ITM calls)
- Put: $-1 \leq \Delta_P \leq 0$ (approaches -1 for deep ITM puts)

### Gamma (Γ)

**Definition**: Rate of change of delta with respect to underlying price.
$$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}$$

**Formula** (same for calls and puts):
$$\Gamma = \frac{e^{-q\tau} n(d_1)}{S \sigma \sqrt{\tau}}$$

where $n(d_1) = \frac{1}{\sqrt{2\pi}} e^{-d_1^2/2}$

**Derivation**:
$$\frac{\partial \Delta_C}{\partial S} = \frac{\partial}{\partial S}\left[e^{-q\tau} N(d_1)\right]$$

$$= e^{-q\tau} n(d_1) \frac{\partial d_1}{\partial S} = e^{-q\tau} n(d_1) \frac{1}{S\sigma\sqrt{\tau}}$$

**Properties**:
- Always positive for long options (convexity)
- Maximum at-the-money (ATM)
- Decreases as option moves ITM or OTM
- Gamma risk: P&L impact of large price moves

### Vega (ν)

**Definition**: Rate of change of option price with respect to volatility.
$$\nu = \frac{\partial V}{\partial \sigma}$$

**Formula** (same for calls and puts):
$$\nu = S e^{-q\tau} n(d_1) \sqrt{\tau}$$

Often scaled to represent dollar change per 1% volatility change:
$$\nu_{\text{scaled}} = 0.01 \times S e^{-q\tau} n(d_1) \sqrt{\tau}$$

**Derivation**:
$$\frac{\partial C}{\partial \sigma} = S e^{-q\tau} n(d_1) \frac{\partial d_1}{\partial \sigma} - K e^{-r\tau} n(d_2) \frac{\partial d_2}{\partial \sigma}$$

Using $\frac{\partial d_1}{\partial \sigma} = -\frac{\ln(S/K) + (r-q+\sigma^2/2)\tau}{\sigma^2\sqrt{\tau}} + \frac{\sqrt{\tau}}{1} = \frac{\partial d_2}{\partial \sigma} + \sqrt{\tau}$

and the spot-strike duality $S e^{-q\tau} n(d_1) = K e^{-r\tau} n(d_2)$:
$$\nu = S e^{-q\tau} n(d_1) \sqrt{\tau}$$

**Properties**:
- Always positive for long options
- Maximum at-the-money
- Longer maturity → higher vega
- Vega risk: exposure to implied volatility changes

### Theta (Θ)

**Definition**: Rate of change of option price with respect to time (time decay).
$$\Theta = \frac{\partial V}{\partial t} = -\frac{\partial V}{\partial \tau}$$

**Call Theta**:
$$\Theta_C = -\frac{S n(d_1) \sigma e^{-q\tau}}{2\sqrt{\tau}} - r K e^{-r\tau} N(d_2) + q S e^{-q\tau} N(d_1)$$

**Put Theta**:
$$\Theta_P = -\frac{S n(d_1) \sigma e^{-q\tau}}{2\sqrt{\tau}} + r K e^{-r\tau} N(-d_2) - q S e^{-q\tau} N(-d_1)$$

Scaled to represent dollar change per day (divide by 365):
$$\Theta_{\text{daily}} = \Theta_{\text{annual}} / 365$$

**Derivation**:
From the Black-Scholes PDE:
$$\Theta + (r-q)S\Delta + \frac{1}{2}\sigma^2 S^2 \Gamma - rV = 0$$

Solving for Θ and substituting known Greeks gives the formula above.

**Properties**:
- Usually negative for long options (time decay)
- Maximum (most negative) at-the-money
- Accelerates as expiration approaches
- Theta-Gamma relationship: $\Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma$

### Rho (ρ)

**Definition**: Rate of change of option price with respect to interest rate.
$$\rho = \frac{\partial V}{\partial r}$$

**Call Rho**:
$$\rho_C = K \tau e^{-r\tau} N(d_2)$$

**Put Rho**:
$$\rho_P = -K \tau e^{-r\tau} N(-d_2)$$

Scaled to represent dollar change per 1% rate change (multiply by 0.01):
$$\rho_{\text{scaled}} = 0.01 \times K \tau e^{-r\tau} N(\pm d_2)$$

**Properties**:
- Call rho is positive (calls benefit from higher rates)
- Put rho is negative (puts benefit from lower rates)
- Longer maturity → higher absolute rho
- Less significant for short-dated options

## Finite Difference Methods

Finite difference methods (FDM) solve the Black-Scholes PDE numerically on a discretized grid. They are essential for American options and complex boundary conditions.

### Grid Setup

**Space discretization** (spot price):
- Grid points: $S_i = i \Delta S$ for $i = 0, 1, ..., M$
- Spacing: $\Delta S = S_{\max} / M$
- Typical: $S_{\max} = 3K$ to $5K$

**Time discretization**:
- Time steps: $t_n = n \Delta t$ for $n = 0, 1, ..., N$
- Step size: $\Delta t = T / N$

**Approximation**: $V(S_i, t_n) \approx V_i^n$

### Finite Difference Approximations

**First derivative** (central difference):
$$\frac{\partial V}{\partial S}\bigg|_{i,n} \approx \frac{V_{i+1}^n - V_{i-1}^n}{2\Delta S}$$

**Second derivative** (central difference):
$$\frac{\partial^2 V}{\partial S^2}\bigg|_{i,n} \approx \frac{V_{i+1}^n - 2V_i^n + V_{i-1}^n}{(\Delta S)^2}$$

**Time derivative** (backward difference):
$$\frac{\partial V}{\partial t}\bigg|_{i,n} \approx \frac{V_i^{n+1} - V_i^n}{\Delta t}$$

### Explicit Euler Method

**Discretized PDE**:
$$\frac{V_i^{n+1} - V_i^n}{\Delta t} + (r-q)S_i \frac{V_{i+1}^n - V_{i-1}^n}{2\Delta S} + \frac{1}{2}\sigma^2 S_i^2 \frac{V_{i+1}^n - 2V_i^n + V_{i-1}^n}{(\Delta S)^2} - rV_i^n = 0$$

**Rearranged**:
$$V_i^{n+1} = a_i V_{i-1}^n + b_i V_i^n + c_i V_{i+1}^n$$

where:
$$a_i = \frac{\Delta t}{2}\left[\frac{1}{2}\sigma^2 i^2 - (r-q)i\right]$$
$$b_i = 1 - \Delta t[\sigma^2 i^2 + r]$$
$$c_i = \frac{\Delta t}{2}\left[\frac{1}{2}\sigma^2 i^2 + (r-q)i\right]$$

**Stability Condition**:
$$\Delta t \leq \frac{(\Delta S)^2}{\sigma^2 S_{\max}^2}$$

If violated, the scheme becomes unstable (oscillations, exponential growth).

### Implicit Euler Method

**Discretized PDE**:
$$\frac{V_i^{n+1} - V_i^n}{\Delta t} + (r-q)S_i \frac{V_{i+1}^{n+1} - V_{i-1}^{n+1}}{2\Delta S} + \frac{1}{2}\sigma^2 S_i^2 \frac{V_{i+1}^{n+1} - 2V_i^{n+1} + V_{i-1}^{n+1}}{(\Delta S)^2} - rV_i^{n+1} = 0$$

**Matrix Form**:
$$\mathbf{A} \mathbf{V}^{n+1} = \mathbf{V}^n$$

where $\mathbf{A}$ is a tridiagonal matrix:
$$-a_i V_{i-1}^{n+1} + (1 + \Delta t[\sigma^2 i^2 + r]) V_i^{n+1} - c_i V_{i+1}^{n+1} = V_i^n$$

**Solution**: Solve the linear system using Thomas algorithm (tridiagonal matrix algorithm).

**Stability**: Unconditionally stable for any $\Delta t$ (but accuracy decreases for large $\Delta t$).

### Crank-Nicolson Method

**Weighted Average**: Combine explicit and implicit schemes (θ-method with θ=0.5):
$$\frac{V_i^{n+1} - V_i^n}{\Delta t} + \frac{1}{2}\left[\mathcal{L}V_i^n + \mathcal{L}V_i^{n+1}\right] = 0$$

where $\mathcal{L}$ is the spatial differential operator.

**Matrix Form**:
$$(I - \frac{\Delta t}{2}\mathbf{L})\mathbf{V}^{n+1} = (I + \frac{\Delta t}{2}\mathbf{L})\mathbf{V}^n$$

**Properties**:
- Second-order accurate in time: $O(\Delta t^2)$ vs $O(\Delta t)$ for Euler
- Unconditionally stable
- Most commonly used for option pricing
- No oscillations (monotone scheme)

**Implementation**:
```python
# Crank-Nicolson coefficient matrix
def build_crank_nicolson_matrix(S_grid, dt, sigma, r, q):
    M = len(S_grid) - 1
    A = np.zeros((M+1, M+1))

    for i in range(1, M):
        a_i = 0.25 * dt * (sigma**2 * i**2 - (r-q) * i)
        b_i = -0.5 * dt * (sigma**2 * i**2 + r)
        c_i = 0.25 * dt * (sigma**2 * i**2 + (r-q) * i)

        A[i, i-1] = -a_i
        A[i, i] = 1 - b_i
        A[i, i+1] = -c_i

    return A
```

### American Option Early Exercise

For American options, at each time step, check early exercise condition:
$$V_i^{n+1} = \max\left(V_i^{\text{continuation}}, V_i^{\text{exercise}}\right)$$

**Put Option**:
$$V_i^{n+1} = \max\left(V_i^{\text{FDM}}, K - S_i\right)$$

**Call Option with Dividends**:
$$V_i^{n+1} = \max\left(V_i^{\text{FDM}}, S_i - K\right)$$

**Optimal Exercise Boundary**: Track the lowest spot price at which early exercise is optimal at each time.

## Monte Carlo Simulation

Monte Carlo methods estimate option prices by simulating many possible price paths and averaging the discounted payoffs.

### Geometric Brownian Motion

**Continuous-time SDE**:
$$dS_t = (r-q)S_t dt + \sigma S_t dW_t$$

**Analytical Solution**:
$$S_t = S_0 \exp\left[(r - q - \frac{\sigma^2}{2})t + \sigma W_t\right]$$

**Discretization** (Euler-Maruyama):
$$S_{t+\Delta t} = S_t \exp\left[(r - q - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} \, Z\right]$$

where $Z \sim N(0,1)$ is a standard normal random variable.

### Basic Monte Carlo Algorithm

1. **Generate** M independent price paths
2. **Calculate** payoff for each path: $\text{Payoff}_i = \max(S_T^{(i)} - K, 0)$ for call
3. **Average** discounted payoffs: $C \approx e^{-rT} \frac{1}{M} \sum_{i=1}^M \text{Payoff}_i$

**Standard Error**:
$$SE = \frac{\sigma_{\text{payoff}}}{\sqrt{M}}$$

where $\sigma_{\text{payoff}}$ is the sample standard deviation of payoffs.

**Confidence Interval** (95%):
$$\text{Price} \pm 1.96 \times SE$$

**Convergence Rate**: $O(M^{-1/2})$ - to halve the error, need 4x simulations.

### Variance Reduction Techniques

#### Antithetic Variates

**Idea**: For each random draw Z, also simulate with -Z.

**Implementation**:
```python
Z = np.random.standard_normal(M // 2)
S_plus = S0 * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
S_minus = S0 * np.exp((r - q - 0.5*sigma**2)*T - sigma*np.sqrt(T)*Z)

payoff_plus = np.maximum(S_plus - K, 0)
payoff_minus = np.maximum(S_minus - K, 0)

price = np.exp(-r*T) * np.mean([payoff_plus, payoff_minus])
```

**Variance Reduction**: Can reduce variance by 50%+ for symmetric payoffs.

#### Control Variates

**Idea**: Use a related variable with known expectation to reduce variance.

**For European Call**: Use geometric average Asian option as control variate.

**Formula**:
$$\hat{C} = C_{\text{MC}} + \beta(C_{\text{control}}^{\text{analytical}} - C_{\text{control}}^{\text{MC}})$$

**Optimal β**:
$$\beta^* = \frac{\text{Cov}(C, C_{\text{control}})}{\text{Var}(C_{\text{control}})}$$

**Variance Reduction**: Up to $1 - \rho^2$ where ρ is the correlation.

### Greeks via Finite Differences

**Delta** (pathwise method):
$$\Delta \approx \frac{C(S_0 + \epsilon) - C(S_0 - \epsilon)}{2\epsilon}$$

Simulate twice: once with $S_0 + \epsilon$, once with $S_0 - \epsilon$.

**Gamma**:
$$\Gamma \approx \frac{C(S_0 + \epsilon) - 2C(S_0) + C(S_0 - \epsilon)}{\epsilon^2}$$

**Vega**:
$$\nu \approx \frac{C(\sigma + \epsilon) - C(\sigma - \epsilon)}{2\epsilon}$$

## Lattice Models

### Binomial Tree (Cox-Ross-Rubinstein)

**Tree Parameters**:
$$u = e^{\sigma\sqrt{\Delta t}}$$
$$d = e^{-\sigma\sqrt{\Delta t}} = \frac{1}{u}$$

**Risk-Neutral Probability**:
$$p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

**Price Evolution**:
- Up move: $S_{i+1,j+1} = u \cdot S_{i,j}$
- Down move: $S_{i+1,j} = d \cdot S_{i,j}$

**Backward Induction**:
1. **Terminal values**: $V_{N,j} = \max(S_{N,j} - K, 0)$ for call
2. **Backward step**:
   $$V_{i,j} = e^{-r\Delta t}[p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}]$$
3. **American**: $V_{i,j} = \max(\text{continuation}, \text{exercise})$

**Convergence**: $O(N^{-1})$ for European, $O(N^{-1/2})$ for American.

### Trinomial Tree

**Branching**:
- Up: $u = e^{\sigma\sqrt{3\Delta t}}$
- Middle: $m = 1$
- Down: $d = e^{-\sigma\sqrt{3\Delta t}}$

**Probabilities**:
$$p_u = \left[\frac{\sigma\sqrt{\Delta t}}{\sqrt{3}} + \frac{(r-q)\Delta t}{\sqrt{3}\sigma\sqrt{\Delta t}}\right]^2$$
$$p_d = \left[\frac{\sigma\sqrt{\Delta t}}{\sqrt{3}} - \frac{(r-q)\Delta t}{\sqrt{3}\sigma\sqrt{\Delta t}}\right]^2$$
$$p_m = 1 - p_u - p_d$$

**Advantages**:
- More flexible than binomial
- Better convergence for barrier options
- Can center tree on current spot price

## Implied Volatility

Implied volatility (IV) is the volatility parameter σ that equates the Black-Scholes theoretical price to the observed market price.

### Problem Formulation

Find σ such that:
$$C_{\text{BS}}(S, K, T, \sigma, r, q) = C_{\text{market}}$$

This is a root-finding problem:
$$f(\sigma) = C_{\text{BS}}(\sigma) - C_{\text{market}} = 0$$

### Newton-Raphson Method

**Iteration**:
$$\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{market}}}{\nu(\sigma_n)}$$

where $\nu(\sigma_n)$ is vega.

**Algorithm**:
1. Initial guess: $\sigma_0 = 0.2$ (20% volatility)
2. Iterate until $|C_{\text{BS}}(\sigma_n) - C_{\text{market}}| < \epsilon$
3. Typical convergence: 3-5 iterations

**Advantages**:
- Fast convergence (quadratic)
- Few iterations needed

**Disadvantages**:
- Requires vega calculation
- Can fail for extreme parameters
- No guaranteed convergence

### Brent's Method

**Hybrid approach**: Combines bisection, secant, and inverse quadratic interpolation.

**Advantages**:
- Guaranteed convergence (bracketing method)
- Super-linear convergence
- Robust for all parameter ranges

**Disadvantages**:
- Slightly slower than Newton-Raphson
- Requires initial bracket [a, b] where $f(a) \cdot f(b) < 0$

**Initial Bracket**:
- Lower bound: σ = 0.001 (near-zero volatility)
- Upper bound: σ = 5.0 (very high volatility)

### Initial Guess Heuristics

**Brenner-Subrahmanyam approximation** (ATM options):
$$\sigma \approx \frac{C_{\text{market}} \sqrt{2\pi}}{S\sqrt{T}}$$

**Corrado-Miller extension** (away from ATM):
$$\sigma \approx \frac{\sqrt{2\pi}}{S\sqrt{T}}\left[C_{\text{market}} - \frac{S - K}{2} + \sqrt{\left(C_{\text{market}} - \frac{S - K}{2}\right)^2 - \frac{(S - K)^2}{\pi}}\right]$$

## Volatility Surface Modeling

### SVI Parameterization

**Stochastic Volatility Inspired (SVI)** model:
$$\sigma_{\text{total}}^2(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \eta^2}\right]$$

where:
- $k = \ln(K/F)$ is log-moneyness
- $F = S e^{(r-q)T}$ is forward price
- Parameters: a, b, ρ, m, η

**Calibration**: Minimize sum of squared errors:
$$\min_{a,b,\rho,m,\eta} \sum_i \left(\sigma_{\text{SVI}}^2(k_i) - \sigma_{\text{market}}^2(k_i)\right)^2$$

Subject to no-arbitrage constraints.

### No-Arbitrage Conditions

**Calendar Arbitrage**: Implied variance must increase with maturity:
$$\sigma_1^2 T_1 < \sigma_2^2 T_2 \quad \text{for } T_1 < T_2$$

**Butterfly Arbitrage**: Call price must be convex in strike:
$$\frac{\partial^2 C}{\partial K^2} \geq 0$$

Equivalent to density being non-negative:
$$g(K) = e^{rT}\frac{\partial^2 C}{\partial K^2} \geq 0$$

## Numerical Accuracy and Stability

### Floating-Point Precision

All calculations use `float64` (IEEE 754 double precision):
- Precision: ~15-17 decimal digits
- Range: $\pm 10^{\pm 308}$
- Machine epsilon: $\epsilon \approx 2.22 \times 10^{-16}$

### Common Numerical Issues

**1. Catastrophic Cancellation**

**Problem**: $\ln(S/K)$ when S ≈ K
**Solution**: Use `np.log1p` for $\ln(1 + x)$ when x is small

**2. Overflow/Underflow in $e^{-rT}$**

**Problem**: Very large or small exponents
**Solution**: Scale calculations, use log-space arithmetic

**3. Normal CDF for extreme values**

**Problem**: $N(x)$ for $|x| > 10$
**Solution**: Use asymptotic approximations or ensure proper library implementation

### Convergence Tests

**Monte Carlo**:
$$\text{Relative Error} = \frac{1.96 \times SE}{C} < 0.01 \quad (1\% \text{ accuracy})$$

**Finite Difference**:
$$|V^{\text{FDM}} - V^{\text{analytical}}| < 10^{-4}$$

**Lattice Models**:
$$|V_N - V_{2N}| < 10^{-3} \quad \text{(extrapolation check)}$$

## References

### Foundational Papers

1. **Black, F., & Scholes, M. (1973)**. "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. **Merton, R. C. (1973)**. "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.

3. **Cox, J. C., Ross, S. A., & Rubinstein, M. (1979)**. "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229-263.

4. **Heston, S. L. (1993)**. "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2), 327-343.

### Numerical Methods

5. **Wilmott, P., Howison, S., & Dewynne, J. (1995)**. *The Mathematics of Financial Derivatives*. Cambridge University Press.

6. **Glasserman, P. (2003)**. *Monte Carlo Methods in Financial Engineering*. Springer.

7. **Duffy, D. J. (2006)**. *Finite Difference Methods in Financial Engineering: A Partial Differential Equation Approach*. Wiley.

### Implied Volatility

8. **Brenner, M., & Subrahmanyam, M. G. (1988)**. "A Simple Formula to Compute the Implied Standard Deviation." *Financial Analysts Journal*, 44(5), 80-83.

9. **Jäckel, P. (2015)**. *Let's Be Rational*. Wilmott Magazine.

### Volatility Surface

10. **Gatheral, J., & Jacquier, A. (2014)**. "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance*, 14(1), 59-71.

---

**Document Version**: 1.0
**Last Updated**: 2025-12-13
**Authors**: Quantitative Research Team
**Review Cycle**: Annually
