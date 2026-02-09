---
id: 45c8eef9
title: "Resolve All Codebase Linting Errors"
status: Triage
priority: High
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [lint, quality]
assignee: Pickle Rick
---

# Description

## Problem to solve
Running `./scripts/lint_all.sh` (or `start_all_dev.sh --lint`) fails due to unresolved ruff errors in the codebase.

## Solution
1. Identify all files with linting errors.
2. Fix auto-fixable errors with `ruff check . --fix`.
3. Manually resolve any remaining errors that `ruff` cannot fix automatically.
4. Verify with a clean run of `scripts/lint_all.sh`.
