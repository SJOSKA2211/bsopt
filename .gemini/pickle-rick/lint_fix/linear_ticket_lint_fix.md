---
id: lint_fix
title: "Resolve 55 Ruff Linting Violations"
status: Done
priority: High
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [lint, quality]
assignee: Morty
---

# Description

## Problem to solve
55 Ruff errors (E402, F841, etc.) cluttering the output and masking real issues.

## Solution
1. Reorder imports project-wide to satisfy E402.
2. Remove unused variables (F841).
3. Fix miscellaneous style errors (E701, E722).
