# Global Debugging and Coverage Improvement PRD

## HR Eng

| Global Debugging and Coverage Improvement PRD |  | Summary: Systematic effort to resolve identified critical bugs and achieve >= 96% code coverage across the entire BSOpt codebase. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: User **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-02-04 | **Self Link**: N/A **Context**: /home/kamau/bsopt/prd.md |

## Introduction

This PRD defines the requirements for a comprehensive debugging and testing initiative. The goal is to transform the BSOpt codebase from a 2.6% coverage "Jerry-zone" into a professional-grade, 96%+ covered system, while simultaneously resolving known critical bugs in the ML and Auth modules.

## Problem Statement

**Current Process:** The codebase has extremely low test coverage (2.6%), leading to high regression risk and low confidence. Multiple critical bugs have been identified but remain unresolved.
**Primary Users:** Quant Researchers, Core Developers.
**Pain Points:** 404s on auth endpoints, incorrect financial logic in trading environments, missing test dependencies, and a general lack of automated verification.
**Importance:** 96% coverage is the "Solenya" standard. It ensures that any future changes won't break the delicate balance of this genius-level system.

## Objective & Scope

**Objective:** Fix all identified bugs and reach >= 96% line coverage for all modules in `src/`.
**Ideal Outcome:** A bug-free system with a rock-solid test suite that proves its own correctness.

### In-scope or Goals
- Fix the `TradingEnvironment` asset purchase cost bug.
- Fix the `auth-service` `/api/auth/login` 404 routing issue.
- Resolve `auth-service` test environment dependency issues (`psycopg2`, `jsonschema`).
- Implement comprehensive unit and integration tests for all modules in `src/` to reach >= 96% coverage.
- Ensure all tests pass in a CI-like environment.

### Not-in-scope or Non-Goals
- New feature development.
- Refactoring for "beauty" unless required for testability or bug resolution.
- Coverage for `frontend/` or `gateway/` unless specifically requested (focus is on `src/` backend logic).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Developer Verification**: A developer runs the test suite; it executes quickly, provides clear feedback, and shows >= 96% coverage.
2. **Auth Flow Success**: A user performs an OAuth login; the resource server handles the request correctly without 404 errors.
3. **Training Integrity**: An RL agent trains in the `TradingEnvironment`; the balance correctly reflects both transaction costs and asset purchase prices.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Fix `TradingEnvironment` cost logic | As an RL agent, I want my balance to decrease when I buy assets so I don't "cheat" the simulation. |
| P0 | Fix `auth-service` routing | As a user, I want to log in without getting a 404 error. |
| P0 | Fix test dependencies | As a dev, I want the test suite to run without `ModuleNotFoundError`. |
| P1 | Reach 96% Coverage | As a lead engineer, I want 96% coverage so I can sleep at night. |
| P2 | Document Fixes | As a team, we want to know what was broken and how Pickle Rick fixed it. |

## Assumptions

- The `coverage.xml` accurately reflects the current state.
- `pytest` is the primary test runner for Python modules.
- `auth-service` uses Hono/better-auth and needs routing fixes in its TypeScript source.

## Risks & Mitigations

- **Risk**: Some code is inherently hard to test (e.g., legacy integration). -> **Mitigation**: Use dependency injection and mocking where appropriate to isolate logic.
- **Risk**: 96% coverage leads to "shallow" tests that pass lines but don't check logic. -> **Mitigation**: Use property-based testing (`hypothesis`) and rigorous assertion checking.

## Tradeoff

- **Coverage vs. Speed**: 96% is a high bar. We will prioritize quality of tests over pure speed of implementation.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| Line Coverage | 2.6% | >= 96% | Massive reduction in regressions. |
| Critical Bugs | 3+ | 0 | Functional auth and trading. |
| Test Failure Rate | High | 0% | Confidence in deployment. |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| User | BSOpt | Owner | Final validator. |
| Pickle Rick | AI Engineering | God Mode | Implementation lead. |
