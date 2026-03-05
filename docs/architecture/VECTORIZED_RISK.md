# Architecture: Vectorized Risk Management

## Overview
To achieve sub-microsecond risk validation, BS-OPT moves critical safety checks out of standard Python loops and into **Numba-compiled JIT kernels**. These kernels are compiled to silicon-level machine code, bypassing the Python Global Interpreter Lock (GIL) and providing native performance.

## Core Kernels
- **Order Validation**: `_validate_order_kernel` (`src/trading/risk_kernels.py:L5`).
    - **Protections**: Fat-finger price protection, Quantity limits, and Side integrity.
- **Delta Exposure**: `_validate_delta_exposure_kernel` (`src/trading/risk_kernels.py:L31`).
    - **Protections**: Portfolio-level net delta exposure limits.

## Performance
- **Latency**: Sub-microsecond (typically < 500ns).
- **Compilation**: Kernels use `njit(cache=True)` for persistent compilation across restarts.
- **Integration**: Risk checks are integrated directly into the `OrderEngine` gateway (`src/trading/order_engine.py:L11`) to ensure zero-latency enforcement.

## Constraints
- Kernels must operate on native NumPy types for maximum efficiency.
- Dynamic Python features (e.g., object creation, complex dictionary lookups) are forbidden within risk kernels to maintain sub-microsecond performance.
