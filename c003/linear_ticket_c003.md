---
id: c003
title: Refactor Neural Greeks DAG
status: Triage
priority: Medium
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [refactor, ml]
assignee: Pickle Rick
---

# Description
## Problem
`dag_neural_greeks.py` needs refactoring to use the new Transformer Policy (audit report).

## Solution
1. Locate `dag_neural_greeks.py`.
2. Refactor to use Transformer Policy.
3. Verify DAG construction.
