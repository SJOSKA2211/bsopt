# Optimize Task Initialization Implementation Plan

## Overview
Reduce per-task overhead in Celery workers by caching `PricingStrategy` instances locally within the worker process, avoiding redundant object creation and initialization.

## Current State Analysis
- **Task Implementation**: `src/tasks/pricing_tasks.py:81` calls `PricingEngineFactory.get_strategy("black_scholes")` on every task execution.
- **Overhead**: Strategy objects, while not massive, still involve allocation and setup that can be avoided.

## Implementation Approach
1. Define a global `_STRATEGY_CACHE` dictionary in `src/tasks/pricing_tasks.py`.
2. Implement a helper function `_get_cached_strategy(name)` that manages the cache.
3. Update `price_option_task` and `batch_price_options_task` to use the cached instances.

## Phase 1: Engine Caching
### Overview
Implement the cache and refactor the task logic.

### Changes Required:

#### 1. src/tasks/pricing_tasks.py
**Changes**: Add cache and update engine acquisition.
```python
_STRATEGY_CACHE = {}

def _get_cached_strategy(name: str):
    """🚀 SINGULARITY: Worker-local engine caching."""
    if name not in _STRATEGY_CACHE:
        _STRATEGY_CACHE[name] = PricingEngineFactory.get_strategy(name)
    return _STRATEGY_CACHE[name]

# In tasks:
# engine = _get_cached_strategy("black_scholes")
```

### Success Criteria:
#### Automated:
- [ ] `pytest tests/integration/workers/test_math_worker.py` passes.
#### Manual:
- [ ] Profile a 1000-task run and confirm reduction in total CPU time compared to baseline.

**Implementation Note**: Requires no new dependencies.
