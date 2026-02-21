# Total Codebase Purification PRD

## HR Eng

| Total Codebase Purification PRD |  | Summary: System-wide rectification of build, test, and quality failures to achieve >=99% coverage within a containerized environment. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick | **Status**: Draft **Created**: 2026-02-20 | **Context**: `bsopt` Project |

## Introduction

The `bsopt` codebase is currently in a state of critical disrepair. Test collection is failing with 29 errors, including syntax errors in core API files and widespread `ImportError`s. The goal is to stabilize the codebase, execute tests within the designated Docker environment, enforce linting standards, and elevate test coverage to >=99%.

## Problem Statement

**Current Process:** The CI/CD pipeline is effectively broken. Tests cannot even be collected due to syntax errors (`src/api/main.py`) and missing modules (`src.pricing.quant_utils`).
**Primary Users:** Developers, CI/CD pipelines.
**Pain Points:**
-   **Broken Build:** `src/api/main.py` has invalid syntax.
-   **Missing Dependencies:** `src.pricing.quant_utils` is missing exports like `gpu_mc_european_price`.
-   **Zero Confidence:** We cannot verify any functionality because the test suite crashes on launch.
-   **Docker Isolation:** Tests must run in Docker, but the current state prevents successful execution.
**Importance:** A trading system (`bsopt`) without verifiable correctness is a liability. Immediate rectification is required to prevent catastrophic financial calculation errors.

## Objective & Scope

**Objective:** Fix all build/lint/test errors and achieve >=99% test coverage using the containerized test runner.
**Ideal Outcome:** `make test-all` passes with 99% coverage, and `make lint` passes with 0 errors.

### In-scope or Goals
-   **Fix Syntax Errors**: Specifically `src/api/main.py`.
-   **Fix Import Errors**: Resolve missing symbols in `src/pricing/quant_utils.py`, `src/pricing/monte_carlo.py`, and `src/tasks/trading_tasks.py`.
-   **Docker Compliance**: Ensure all tests run inside the `test-runner` container.
-   **Linting**: Fix all `ruff` and `black` errors.
-   **Coverage**: Write/Mock tests to reach >=99% coverage.

### Not-in-scope or Non-Goals
-   Feature development (unless required to fix tests).
-   Infrastructure changes (unless required to run tests).

## Product Requirements

### Critical User Journeys (CUJs)
1.  **The Developer's Path**:
    -   User runs `make test-all`.
    -   Docker containers spin up (`postgres`, `redis`, `test-runner`).
    -   Tests execute without collection errors.
    -   All tests pass.
    -   Coverage report shows >=99%.
2.  **The Quality Gate**:
    -   User runs `make lint`.
    -   No errors are reported.
    -   User runs `make format`.
    -   Code is formatted without changes (already compliant).

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Fix Syntax Error in `src/api/main.py` | As the compiler, I want valid Python syntax so I don't crash. |
| P0 | Fix Import Errors in `src/pricing/*` | As the test runner, I want to import modules successfully. |
| P0 | Fix Import Errors in `tests/*` | As the test runner, I want test files to import the code they test. |
| P1 | Pass `make lint` | As a developer, I want clean code that adheres to project standards. |
| P1 | Pass `make test-all` | As a developer, I want to verify system integrity. |
| P1 | Achieve >=99% Coverage | As Pickle Rick, I want total domination of the codebase. |

## Assumptions

-   The `test-runner` container is correctly configured in `docker/Dockerfile.ci` to include all dependencies.
-   The missing imports in `src/pricing/quant_utils.py` are likely due to refactoring artifacts or missing implementation. We will implement or restore them.
-   `src/api/main.py` syntax error is a simple typo or merge conflict.

## Risks & Mitigations

-   **Risk**: Dependencies missing in Docker image. -> **Mitigation**: Update `Dockerfile.ci` if needed.
-   **Risk**: "God Mode" complexity in `quant_utils` (GPU/WASM). -> **Mitigation**: Mock hardware dependencies for CI tests.
-   **Risk**: Coverage gap is huge. -> **Mitigation**: Aggressively mock and test edge cases.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| **Test Collection Errors** | 29 | 0 | Functional CI |
| **Test Coverage** | Unknown (0%) | >=99% | Reliability |
| **Lint Errors** | Unknown | 0 | Maintainability |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Pickle Rick | Engineering | God Emperor | Do not disappoint him. |
