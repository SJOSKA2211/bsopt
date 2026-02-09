# Global Debugging and Coverage Improvement PRD (Pickle Rick Edition)

## Overview
This Product Requirements Document (PRD) outlines the comprehensive initiative to stabilize, debug, and achieve >=97% code coverage for the `bsopt` codebase. This effort is critical for enhancing code quality, reliability, and developer efficiency.

## Problem Statement
The current `bsopt` codebase exhibits:
1.  **Low Code Coverage:** Significantly below industry best practices, leading to undetected bugs and regressions.
2.  **Test Collection Errors:** Multiple critical errors preventing full test suite execution, hindering effective quality assurance.
3.  **Stability Issues:** Identified bugs and inconsistencies across core components (Auth, RL, DB, ML pipeline).
4.  **Developer Friction:** The unreliable testing environment and existing bugs impede rapid development and feature delivery.

## Goals
- Achieve >=97% code coverage across all core `bsopt` modules.
- Eliminate all test collection errors, allowing the full test suite to run successfully.
- Resolve critical bugs and optimize key components as identified in the `task_state`.
- Establish a robust and reliable testing framework.

## Scope
### In-Scope
- Resolution of all `ImportError` and `SyntaxError` issues in test collection.
- Implementation of necessary code aliases (e.g., GNN classes).
- Fixing the `IndentationError` in `src/services/pricing_service.py` (already addressed in working tree).
- Systematically increasing unit and integration test coverage.
- Debugging and fixing known application-level bugs (e.g., `TradingEnvironment` asset purchase cost bug, `auth-service` 404 login issue, Kafka/Redis test environment dependencies).
- Refactoring and optimizing the ML pipeline for performance and reliability.
- **Backlog Liquidation & Slop Removal**: This specific task, while important for codebase hygiene, will be handled as a sub-task under the broader coverage improvement goal.

### Out-of-Scope
- Major architectural redesigns not directly related to debugging or coverage.
- Implementation of new features not outlined in existing tasks.

## Key Deliverables
1.  **Fully Executable Test Suite:** All `pytest` collection and execution errors resolved.
2.  **Code Coverage Report:** Demonstrating >=97% coverage.
3.  **Stabilized Core Components:** Verified through passing tests and functional validation.
4.  **Cleaned Codebase:** Removal of "AI Slop" and irrelevant content (subsumed under this PRD's larger goal).

## Success Metrics
- **Code Coverage:** >=97% (measured via `pytest --cov`).
- **Test Pass Rate:** 100% for all existing and newly added tests.
- **Zero Test Collection Errors:** `pytest --collect-only` runs without errors.
- **Critical Bugs Resolved:** All identified critical bugs are fixed and verified.

## Timeline (Iterative Approach)
This will be an iterative process, guided by the Rick Loop.
- **Phase 1 (Current):** Resolve all test collection blockers (`c000_fix`).
- **Phase 2:** Address remaining critical bugs.
- **Phase 3:** Systematically increase test coverage.
- **Phase 4:** Optimize ML pipeline and address remaining known issues.

## Dependencies
- Active Python 3.13.0 environment.
- `pytest` and `pytest-cov` for testing and coverage measurement.
- Access to project codebase and Docker setup for environment replication.

## Stakeholders
- Pickle Rick (Agent)
- Morty (Worker Agents)
- User (Human Developer)

## Risks
- **Rogue Worker Actions:** Mortys may deviate from instructions, requiring frequent reconciliation.
- **Interconnected Bugs:** Fixing one bug may expose others.
- **Environmental Drift:** Inconsistencies between local and CI environments.

## Mitigation
- Clear, atomic task definitions for workers.
- Incremental commits and frequent verification.
- Continuous monitoring of test results.
- Automated environment checks.
