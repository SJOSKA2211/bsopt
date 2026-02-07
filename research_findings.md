# Research: Consolidate Black-Scholes Math

**Date**: 2026-02-06

## 1. Executive Summary
The codebase contains multiple implementations of Black-Scholes pricing and greeks calculation. The goal is to consolidate these into `src/shared/math_utils.py`. Currently, `src/shared/math_utils.py` contains basic building blocks (CDF, PDF, d1/d2), but higher-level pricing and greek functions are scattered.

## 2. Technical Context

### Existing Central Utility
- **File**: `src/shared/math_utils.py`
- **Current Content**:
  - `fast_normal_cdf` (Vectorized)
  - `fast_normal_pdf` (Vectorized)
  - `calculate_d1_d2` (Vectorized)
  - `calculate_d1_d2_scalar` (Scalar)

### Redundant Implementations

1.  **`src/pricing/black_scholes.py`**
    -   Implements `_norm_cdf`, `_norm_pdf` locally (duplicates `math_utils`).
    -   `_price_options_jit`: Core pricing logic using local helpers.
    -   `_calculate_greeks_jit`: Core greeks logic using local helpers.
    -   `BlackScholesEngine`: Wrapper class.

2.  **`src/ml/training/data_gen.py`**
    -   `_black_scholes_numba_kernel`: Standalone kernel for data generation. Re-implements CDF/PDF logic inline.

3.  **`src/pricing/wasm_engine.py`**
    -   References `batch_price_black_scholes`. Needs check if it relies on internal logic or imports.

## 3. Findings & Analysis
- `src/pricing/black_scholes.py` relies on `scipy.stats.norm` for non-JIT contexts (imports it), but defines JIT-compiled versions `_norm_cdf` and `_norm_pdf`.
- `src/shared/math_utils.py` uses `math.erf` for `fast_normal_cdf` which is JIT-compatible.
- There is no centralized `calculate_price` or `calculate_greeks` in `math_utils.py`.
- The `njit` decorators in `math_utils.py` use `cache=True, fastmath=True`.

## 4. Technical Constraints
- Must maintain `numba` compatibility (nopython mode).
- Must support both scalar (float) and vector (numpy array) inputs efficiently.
- Existing `shared/math_utils.py` handles import errors for `numba` gracefully (mocking), which must be preserved.
