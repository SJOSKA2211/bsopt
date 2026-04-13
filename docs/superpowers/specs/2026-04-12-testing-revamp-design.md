# Design Spec: BSOPT Full-Stack Testing Revamp

**Date**: 2026-04-12
**Topic**: Full-Stack Testing & Coverage Optimization

## 1. Overview
The goal is to achieve 100% passing rate and maximum possible coverage across all test tiers (Unit, Integration, E2E) for the BSOPT system. This includes handling the hybrid Python/Rust backend and the React/Playwright frontend.

## 2. Requirements & Constraints
- **100% Passing**: All existing and new tests must pass.
- **Max Coverage**: Aim for 100% logic coverage in critical mathematical and financial kernels.
- **Security Check Compatibility**: Bypass or satisfy Pydantic's 32-character password requirement for testing.
- **Dependency Isolation**: Mock large ML/Distributed dependencies if not available in the local environment.
- **Infrastructure**: Must ensure core services (Postgres, Redis, RabbitMQ) are available during integration and E2E runs.

## 3. Architecture & Data Flow
- **Unit Tier**: Isolated tests with mocks for external services. Fast execution.
- **Integration Tier**: Tests against live Docker containers. Verifies database schemas, cache logic, and message publishing.
- **E2E Tier**: Playwright tests simulating user interactions on the UI, flowing down to the API and worker layers.

## 4. Implementation Strategy
1.  **Environment Hardening**: Update `conftest.py` and `pytest.ini` to enforce a stable test environment.
2.  **Dependency Management**: Install missing optional dependencies (`ml` group) or ensure mocks cover all missing imports.
3.  **Infrastructure Orchestration**: Use `docker compose` to spin up required services before integration/E2E runs.
4.  **Test Refactoring**: Fix broken tests in `tests/unit/api/` and `src/quant/` that are currently failing due to stale mocks or schema changes.
5.  **Coverage Aggregation**: Use `pytest-cov` and `coverage.py` to aggregate results across Python and Rust (via specialized reporters).

## 5. Success Criteria
- [ ] 100% passing tests in `pytest`.
- [ ] 100% passing E2E tests in Playwright.
- [ ] Minimal "ModuleNotFoundError" during collection.
- [ ] Detailed coverage report showing >90% coverage for core logic.

## 6. Self-Review Notes
- No placeholders left (e.g. TBD).
- Internal consistency: Architecture matches implementation plan.
- Scope: Focused on testing only.
