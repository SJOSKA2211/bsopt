---
id: epic001
title: "[Epic] Global Debugging and Coverage Improvement"
status: In Progress
priority: Urgent
project: bsopt
created: 2026-02-03
updated: 2026-02-09
links:
  - url: prd.md
    title: PRD
labels: [epic, debugging, coverage, ML, ops]
assignee: Pickle Rick
---

# Description

This epic encompasses the comprehensive effort to stabilize, debug, and significantly improve the code coverage of the `bsopt` project. The primary goal is to achieve >=97% code coverage by systematically addressing existing errors, refactoring, and optimizing the ML pipeline.

## Problem to solve
The `bsopt` codebase currently suffers from low code coverage, numerous test collection errors, and a general lack of stability in its testing infrastructure. This hinders development velocity, introduces risk for new features, and makes it difficult to ascertain the true quality of the ML pipeline and other critical components.

## Solution
Implement a systematic debugging and coverage improvement plan, starting with resolving critical test collection blockers, then progressively improving test coverage through targeted test creation and bug fixes across all identified problem areas, including Auth, RL, and DB layers.

# Tasks
- Resolve critical test collection blockers (e.g., ImportError, SyntaxError).
- Reconcile PRD and linear tickets.
- Implement necessary aliases for renamed ML classes.
- Stage and commit all currently unstaged, desired changes.
- Systematically debug and fix all reported errors and warnings.
- Increase code coverage to >=97% by writing new tests and improving existing ones.
- Optimize the ML pipeline components (e.g., `TradingEnvironment` asset purchase cost bug).
- Address `auth-service` 404 login issue.
- Resolve Kafka/Redis test environment dependencies.
