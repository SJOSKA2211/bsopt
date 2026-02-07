---
id: e005
title: Test Suite: ML & Pricing
status: Triage
priority: High
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [tests, coverage, ml]
assignee: Pickle Rick
---

# Description
## Problem
ML and Pricing logic is untested.

## Solution
1. Create `tests/test_ml/` and `tests/test_pricing/`.
2. Mock heavy ML/GPU calls.
3. Verify pricing math.
4. Achieve high coverage.
