# Architecture: Vectorized Risk Management

## Overview
To achieve sub-microsecond risk validation, BS-OPT moves critical safety checks out of standard Python loops and into **Numba-compiled JIT kernels**. These kernels are compiled to silicon-level machine code, bypassing the Python Global Interpreter Lock (GIL) and providing native performance.

## Core Kernels
- **Order Validation**: `_validate_order_kernel` (`src/trading/risk_kernels.py`).
    - **Protections**: Fat-finger price protection, Quantity limits, and Side integrity.
- **Multi-Point Risk**: `_full_risk_check_v2_kernel` (`src/trading/risk_kernels.py`).
    - **Protections**: Portfolio-level **Delta, Gamma, and Vega** exposure limits validated in a single fused pass.
- **Incremental Delta**: `_validate_delta_kernel` (`src/trading/risk_kernels.py`).
    - **Protections**: O(1) stateful delta tracking.

## Performance
- **Latency**: Sub-300ns (typically < 300ns for fused checks).
- **Compilation**: Kernels use `njit(cache=True, fastmath=True)` for persistent compilation and SIMD optimization.
- **Integration**: Risk checks are integrated directly into the `OrderEngine` hot loop and the `OrderExecutor` to ensure zero-latency enforcement before any transaction hits the network.

## Atomic State
- **Redis Sync**: Distributed risk state is synchronized atomically via Redis LUA scripts (`ADVANCED_RISK_MATRIX`) to ensure safety across multi-node deployments.
- **SHM Feedback**: Local hot-state is maintained in Shared Memory for the fastest possible read-path.

## Constraints
- Kernels must operate on native NumPy types for maximum efficiency.
- Dynamic Python features (e.g., object creation, complex dictionary lookups) are strictly forbidden within risk kernels to maintain predictable, sub-microsecond execution.
