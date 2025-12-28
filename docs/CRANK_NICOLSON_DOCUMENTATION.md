# Crank-Nicolson Finite Difference Method for Black-Scholes PDE

## Mathematical Foundation

### The Black-Scholes PDE

The Black-Scholes partial differential equation governs the price V(S,t) of a European option:

```
∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0
```

Where:
- V(S,t): Option value as a function of spot price S and time t
- σ: Volatility (annualized standard deviation)
- r: Risk-free interest rate (continuously compounded)
- q: Dividend yield (continuously compounded)
- S: Underlying asset price
- t: Time

### Boundary and Terminal Conditions

**Terminal Condition (t=T):**
```
Call:  V(S,T) = max(S - K, 0)
Put:   V(S,T) = max(K - S, 0)
```

**Spatial Boundaries:**

At S = 0:
```
Call:  V(0,t) = 0
Put:   V(0,t) = K*exp(-r*τ)  where τ = T - t
```

At S → ∞:
```
Call:  V(S,t) → S - K*exp(-r*τ)
Put:   V(S,t) → 0
```

## Numerical Discretization

### Grid Structure

**Spatial Grid:**
- Domain: [S_min, S_max] where S_min = 0.01, S_max = 3*K
- Grid points: S_i = S_min + i*dS for i = 0, 1, ..., M
- Spacing: dS = (S_max - S_min) / M

**Temporal Grid:**
- Domain: [0, T]
- Time steps: t_n = n*dt for n = 0, 1, ..., N
- Spacing: dt = T / N
- Direction: Backward from T to 0 (solving backward in time)

### Finite Difference Approximations

**Central Difference for First Derivative:**
```
∂V/∂S ≈ (V_{i+1} - V_{i-1}) / (2*dS)
Truncation error: O(dS²)
```

**Central Difference for Second Derivative:**
```
∂²V/∂S² ≈ (V_{i+1} - 2*V_i + V_{i-1}) / (dS²)
Truncation error: O(dS²)
```

**Forward Difference for Time:**
```
∂V/∂t ≈ (V^{n-1}_i - V^n_i) / dt
Truncation error: O(dt)
```

### The Crank-Nicolson Scheme

The Crank-Nicolson method applies the **trapezoidal rule** in time, averaging the spatial operators at time levels n and n-1:

```
(V^{n-1}_i - V^n_i)/dt = 0.5 * [L(V^{n-1}) + L(V^n)]
```

Where L is the spatial operator:
```
L(V_i) = 0.5*σ²*S_i²*∂²V/∂S² + (r-q)*S_i*∂V/∂S - r*V_i
```

This results in a **second-order accurate** scheme in both time and space with truncation error O(dt² + dS²).

### Coefficient Derivation

For grid point i, define:
```
S_i = S_min + i*dS
```

**Coefficient for Lower Diagonal (α_i):**
```
α_i = 0.25 * dt * [σ²*S_i²/dS² - (r-q)*S_i/(2*dS)]
```

**Coefficient for Main Diagonal (β_i):**
```
β_i = -0.5 * dt * [σ²*S_i²/dS² + r]
```

**Coefficient for Upper Diagonal (γ_i):**
```
γ_i = 0.25 * dt * [σ²*S_i²/dS² + (r-q)*S_i/(2*dS)]
```

### Matrix Formulation

The discretized system becomes:
```
A * V^{n-1} = B * V^n
```

**Implicit Matrix A:**
```
A = tridiag(-α, 1-β, -γ)
```

**Explicit Matrix B:**
```
B = tridiag(α, 1+β, γ)
```

Both are tridiagonal matrices of size (M+1) × (M+1).

## Stability Analysis

### Unconditional Stability

The Crank-Nicolson scheme for the heat equation (and Black-Scholes PDE) is **unconditionally stable**.

**Proof sketch via von Neumann analysis:**

For the heat equation ∂u/∂t = κ*∂²u/∂x², the amplification factor is:
```
G = (1 - r*sin²(θ/2)) / (1 + r*sin²(θ/2))
```

where r = κ*dt/dx² and θ is the Fourier mode.

For all r > 0 and θ:
```
|G| ≤ 1
```

This guarantees stability for any choice of dt and dS.

**Practical implications:**
- No CFL-type condition required
- Can use larger time steps compared to explicit methods
- Grid refinement driven by accuracy, not stability

### Mesh Ratio

The mesh ratio is defined as:
```
r = dt / dS²
```

For comparison, the explicit Forward Euler method requires:
```
r ≤ 1 / (2*σ²*S_max²)  for stability
```

Crank-Nicolson has no such restriction.

## Accuracy and Convergence

### Truncation Error

**Spatial discretization:** O(dS²)
- Central differences are second-order accurate

**Temporal discretization:** O(dt²)
- Trapezoidal rule (θ=0.5) is second-order accurate

**Overall truncation error:** O(dt² + dS²)

### Consistency

As dt → 0 and dS → 0, the discrete scheme converges to the continuous PDE.

**Verification:**
```
lim_{dt,dS→0} (discrete equation) = (continuous PDE)
```

### Convergence Rate

By the Lax Equivalence Theorem (for well-posed linear problems):
```
Consistency + Stability ⟹ Convergence
```

Expected convergence rate:
```
Error ≈ C₁*dt² + C₂*dS²
```

For balanced grids with dt ∝ dS, the error scales as O(dS²).

## Implementation Details

### Sparse Matrix Storage

The matrices A and B are tridiagonal, requiring only O(M) storage instead of O(M²).

Using scipy.sparse.diags with format='csr':
- Storage: 3*M elements
- Matrix-vector product: O(M) operations
- Solve: O(M) operations (Thomas algorithm)

### Time-Stepping Algorithm

```
1. Initialize: V^N = payoff(S)
2. Build matrices A and B
3. For n = N, N-1, ..., 1:
     a. Compute RHS: b = B * V^n
     b. Enforce boundary conditions on b
     c. Solve: A * V^{n-1} = b
     d. Update: V = V^{n-1}
4. Interpolate V(S₀, 0)
```

### Boundary Condition Enforcement

At each time step, the boundary values are enforced:

**Lower boundary (i=0):**
```
Call: b[0] = 0
Put:  b[0] = K*exp(-r*τ)
```

**Upper boundary (i=M):**
```
Call: b[M] = S_max - K*exp(-r*τ)
Put:  b[M] = 0
```

The matrix rows are modified to identity:
```
A[0,:] = [1, 0, 0, ..., 0]
A[M,:] = [0, 0, ..., 0, 1]
```

## Greeks Calculation

Greeks are computed using finite differences on the solution grid:

### Delta: ∂V/∂S

**Central difference:**
```
Δ ≈ (V(S+h) - V(S-h)) / (2h)
```

where h = dS (grid spacing).

**Accuracy:** O(dS²)

### Gamma: ∂²V/∂S²

**Central difference:**
```
Γ ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
```

**Accuracy:** O(dS²)

### Vega: ∂V/∂σ

**Bump and recompute:**
```
ν ≈ (V(σ+ε) - V(σ-ε)) / (2ε)
```

where ε = 0.01 (1% volatility shift).

Requires solving the PDE twice with perturbed volatility.

### Theta: ∂V/∂t

**Finite difference in time:**
```
Θ ≈ (V(t-δt) - V(t)) / δt
```

where δt = 1/365 (one day).

Requires solving the PDE with reduced maturity.

### Rho: ∂V/∂r

**Bump and recompute:**
```
ρ ≈ (V(r+ε) - V(r-ε)) / (2ε)
```

where ε = 0.0001 (1 basis point).

Requires solving the PDE twice with perturbed rate.

## Performance Characteristics

### Computational Complexity

**Per time step:**
- Matrix-vector product: O(M)
- Sparse solve: O(M) (Thomas algorithm for tridiagonal)

**Total:**
- Time complexity: O(M*N)
- Space complexity: O(M)

**Greeks calculation:**
- Delta, Gamma: O(1) (use existing solution)
- Vega, Theta, Rho: O(M*N) each (require new solves)

### Typical Performance

For M=100, N=100:
- Single solve: ~1-5 ms
- Greeks: ~10-20 ms (3 additional solves)

For M=200, N=500:
- Single solve: ~10-50 ms
- Greeks: ~100-200 ms

## Numerical Verification

### Validation Tests

**1. Zero maturity:**
```
V(S,0) = max(S-K, 0)  for calls
V(S,0) = max(K-S, 0)  for puts
```

**2. Deep ITM:**
```
Call: V(S,t) ≈ S - K*exp(-r*τ)  for S >> K
Put:  V(S,t) ≈ K*exp(-r*τ) - S  for S << K
```

**3. Put-Call Parity:**
```
C(S,t) - P(S,t) = S*exp(-q*τ) - K*exp(-r*τ)
```

**4. Grid Convergence:**
As M, N → ∞, the solution should converge to analytical values (when available).

### Convergence Study

Refine the grid and verify:
```
|V_fine - V_coarse| → 0  as dS, dt → 0
```

Expected behavior:
```
Error(dS/2, dt/2) ≈ Error(dS, dt) / 4
```

(quadratic convergence due to second-order scheme)

## Comparison with Other Methods

| Method | Stability | Time Accuracy | Space Accuracy | Cost per step |
|--------|-----------|---------------|----------------|---------------|
| Explicit (Forward Euler) | Conditional (dt ≤ c*dS²) | O(dt) | O(dS²) | O(M) |
| Implicit (Backward Euler) | Unconditional | O(dt) | O(dS²) | O(M) |
| **Crank-Nicolson** | **Unconditional** | **O(dt²)** | **O(dS²)** | **O(M)** |

**Advantages of Crank-Nicolson:**
- Best accuracy for given grid size
- Unconditionally stable (like implicit)
- No oscillations (for well-posed problems)
- Industry standard for option pricing

**Disadvantages:**
- Slightly more complex than explicit methods
- Requires linear system solve (but fast for tridiagonal)

## Advanced Topics

### Non-uniform Grids

For better accuracy near critical points (e.g., S=K), use:
```
S_i = S_min * (S_max/S_min)^(i/M)  (geometric spacing)
```

or adaptive grids based on solution gradients.

### American Options

Requires free boundary treatment:
```
V^{n-1}_i = max(V_computed, payoff_i)
```

at each time step (early exercise constraint).

### Multi-dimensional PDEs

For basket options or stochastic volatility:
- Extend to 2D or 3D grids
- Use Alternating Direction Implicit (ADI) methods
- Computational cost: O(M^d * N) where d is dimension

## References

1. **Wilmott, Howison, Dewynne** (1995). *The Mathematics of Financial Derivatives*. Cambridge University Press.

2. **Crank & Nicolson** (1947). "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Proceedings of the Cambridge Philosophical Society*, 43(1), 50-67.

3. **Morton & Mayers** (2005). *Numerical Solution of Partial Differential Equations*. Cambridge University Press.

4. **Duffy** (2006). *Finite Difference Methods in Financial Engineering*. Wiley.

5. **Tavella & Randall** (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.

## File Location

**Implementation:** `/home/kamau/comparison/src/pricing/finite_difference.py`

This file contains the complete, production-ready implementation of the Crank-Nicolson finite difference solver with all mathematical rigor, stability guarantees, and accuracy verification.
