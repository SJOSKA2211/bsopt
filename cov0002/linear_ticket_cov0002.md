---
id: cov0002
title: Coverage Injection
status: Triage
priority: High
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [testing, coverage]
assignee: Pickle Rick
---

# Description

## Problem to solve
Coverage is likely below 97%. We need to systematically add tests for uncovered paths.

## Solution
1.  Analyze `coverage.xml` / report.
2.  Identify files with low coverage.
3.  Write robust unit/integration tests for those files.
4.  Repeat until total coverage >= 97%.

# Tasks
- [ ] Achieve 97% coverage.
