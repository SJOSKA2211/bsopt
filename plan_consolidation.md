# Plan: Consolidate Black-Scholes Math

## Overview
Move core Black-Scholes pricing and greeks logic to `src/shared/math_utils.py` to eliminate duplication and ensure consistent JIT optimization across the codebase.

## Phase 1: Extend Math Utils
### Overview
Add pricing and greeks functions to `src/shared/math_utils.py`.

### Changes Required:
#### 1. `src/shared/math_utils.py`
**Changes**: Add `calculate_price` and `calculate_greeks`.
```python
@njit(cache=True, fastmath=True)
def calculate_price(S, K, T, sigma, r, q, is_call):
    # Use calculate_d1_d2
    # Use fast_normal_cdf
    # Return price
```
```python
@njit(cache=True, fastmath=True)
def calculate_greeks(S, K, T, sigma, r, q, is_call):
    # Use calculate_d1_d2
    # Use fast_normal_pdf, fast_normal_cdf
    # Return tuple of greeks
```

## Phase 2: Refactor Consumers
### Overview
Update existing modules to use the new shared utilities.

### Changes Required:
#### 1. `src/pricing/black_scholes.py`
**Changes**: Remove local JIT functions (`_norm_cdf`, `_price_options_jit`, etc.) and import from `src.shared.math_utils`.

#### 2. `src/ml/training/data_gen.py`
**Changes**: Replace `_black_scholes_numba_kernel` with `calculate_price` from `math_utils`.

## Phase 3: Verification
### Overview
Ensure tests pass and parity is maintained.

### Success Criteria:
- [ ] Run existing tests: `pytest src/pricing/black_scholes.py` (if any) or related tests.
- [ ] Verify `verify_put_call_parity` still works.
