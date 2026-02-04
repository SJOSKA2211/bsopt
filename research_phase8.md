# Research: Singularity Phase 8 (Tasks & Shared Utilities)

**Date**: 2026-02-04

## 1. Executive Summary
Audit of the background tasks and mathematical utilities identifies "Object Creation Slop" in the Celery worker path. The tasks frequently re-initialize engine strategies and perform manual result formatting loops, negating some of the performance benefits of the underlying JIT kernels.

## 2. Technical Context
- **Pricing Task**: `src/tasks/pricing_tasks.py:22` implements single-option pricing.
- **Batch Task**: `src/tasks/pricing_tasks.py:115` implements vectorized batch pricing.
- **Math Utilities**: `src/shared/math_utils.py:1` provides JIT-compiled kernels.

## 3. Findings & Analysis
- **Initialization Overhead**: `price_option_task` calls `PricingEngineFactory.get_strategy` on every execution (`pricing_tasks.py:81`). We should use a cached or singleton-like approach for the engine instances within the worker process.
- **Formatting Slop**: `batch_price_options_task` (`pricing_tasks.py:155`) manually builds a list of dictionaries. This is slow for large batches (> 10,000 items). Vectorized operations should stay in NumPy/Pandas as long as possible before serialization.
- **Shared Slop**: `math_utils.py` has good kernels, but `calculate_d1_d2` (`math_utils.py:34`) uses `np.sqrt` which can be slightly slower than `math.sqrt` for scalar inputs if called frequently in loops.

## 4. Technical Constraints
- Celery tasks must remain serializable.
- Worker memory usage must be monitored for large batch tasks (already partially handled by `gc.collect()`)

## 5. Architecture Documentation
- **Pattern**: Task-based distribution via Celery/RabbitMQ.
- **Optimization Strategy**: Minimizing per-task overhead and maximizing batch throughput.
EOF
