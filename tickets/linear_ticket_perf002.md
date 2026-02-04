---
id: perf002
title: Fix Pricing Performance Regressions
status: Triage
priority: High
project: BS-OPT
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, performance, pricing]
assignee: Pickle Rick
---

# Description

## Problem to solve
`tests/functional/test_performance.py` is failing. Pricing latency is violating SLA.

## Solution
Profile the pricing endpoint. Optimize the bottleneck. Verify with load tests.
