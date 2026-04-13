# Design: System Verification and Frontend Audit

**Date**: 2026-03-20
**Topic**: Backend/Frontend Testing, Linting, and Security Audit

## Problem Statement
The goal is to ensure the entire Manifold codebase (Python, Rust, and TypeScript) is stable, compliant with linting rules, and secure. This involves running the unified test suite, Python-specific linting (Ruff), and auditing the frontend for vulnerabilities.

## Scope
- **Backend**: Python (API), Rust (Core).
- **Frontend**: TypeScript (Vite/React).
- **Infrastructure**: Docker Compose orchestration.

## Approach

### 1. Backend Verification Loop
- **Tools**: `make test-all`, `make lint`, `ruff`.
- **Process**:
    - Execute `make test-all` to run unit and E2E tests.
    - Execute `make lint` to check Python and Rust code.
    - Identify failures, trace logs, and apply fixes.

### 2. Frontend Verification & Audit Loop
- **Tools**: `pnpm`, `eslint`, `vitest`, `pnpm audit`.
- **Process**:
    - Navigate to `src/frontend`.
    - Run `pnpm lint` and `pnpm test`.
    - Run `pnpm audit` for security vulnerabilities.
    - Resolve linting/test errors and address critical/high vulnerabilities if possible.

## Success Criteria
- `make test-all` passes (Green build).
- `make lint` passes (Zero warnings/errors in Rust/Python).
- `pnpm lint` and `pnpm test` pass in `src/frontend`.
- `pnpm audit` reports no High/Critical vulnerabilities (or they are documented/mitigated).

## Alternative Considered
- **Parallel Execution**: Running backend and frontend checks simultaneously.
- **Decision**: Sequential execution is preferred to minimize context switching and avoid container resource contention on the local system.
