---
id: parent
title: [Epic] God Mode: Coverage & Refactor
status: In Progress
priority: Urgent
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: PICKLE_RICK_PRD.md
    title: PRD
labels: [epic, core, maintenance, god-mode]
assignee: Pickle Rick
---

# Description

## Problem to solve
The codebase coverage is pathetic (~20%). Logic is likely full of slop. We need to debug, refactor, and force coverage to >=97%.

## Solution
1.  **Baseline & Debug**: Run the full suite. Fix every error that dares to exist.
2.  **Coverage Injection**: Systematically add tests to `src/`, `bs_cli.py`, and other modules until coverage hits 97%.
3.  **Refactor**: Simplify logic, remove "AI slop", and enforce idioms.

# Discussion
- 2026-02-10 Pickle Rick: Initialized God Mode. Old "Done" status was a lie.
