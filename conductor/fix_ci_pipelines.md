# CI/CD Pipeline Fix Plan

## Objective
Resolve simultaneous GitHub Actions workflow failures across the Master Orchestration, MLOps, and CI/CD pipelines.

## Root Cause Analysis
Based on a codebase scan, the simultaneous failures (and corresponding annotations) are caused by recently introduced linting violations and a critical missing import that causes runtime crashes in the API health checks and dependent services.

1. **`src/database/__init__.py`**: Missing `import asyncio`. This triggers a Ruff `F821` (Undefined name) annotation and causes a runtime `NameError` during database health checks, which crashes the `make test-all` job and any container startup scripts that verify database readiness.
2. **`src/shared/math_utils.py`**: Unused `import os`. Triggers Ruff `F401`.
3. **`src/scrapers/engine.py`**: Unused `import numpy as np` and `import orjson`. Triggers Ruff `F401`.

Because GitHub Actions surfaces annotations globally across a commit, these four errors manifest as annotations across the different failing workflows, and the `test-and-lint` job strictly enforces them, failing the build.

## Implementation Steps

### Step 1: Fix `src/database/__init__.py`
*   Add `import asyncio` to the top of the file to resolve the `F821` error and fix the `health_check()` coroutine runtime crash.

### Step 2: Fix `src/shared/math_utils.py`
*   Remove the unused `import os` statement at the top of the file.

### Step 3: Fix `src/scrapers/engine.py`
*   Remove `import numpy as np` and `import orjson` from the import block, as they are no longer used in the refactored logic.

## Verification & Testing
1.  **Linting**: Run `make lint` and `make format` to verify that all Ruff `F821` and `F401` errors are eliminated.
2.  **Unit Tests**: Run `pytest tests/api/test_main_routes.py` (or `make test-all`) to ensure that the `test_health_endpoints` passes successfully without throwing a `NameError`.
3.  **Pipeline Simulation**: Run `docker compose --profile ml build api` and `docker compose --profile test run --rm test-runner ruff check .` locally to ensure the containerized steps succeed.
