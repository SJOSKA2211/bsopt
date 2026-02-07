---
id: bc423913
title: Research OOM & Fix Environment
status: Done
priority: Urgent
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, infra]
assignee: Pickle Rick
---

# Description

## Problem to solve
Running `pytest` causes Exit Code 137 (OOM). The environment is unstable.

## Solution
1. Profile memory usage during tests.
2. Identify leaks (likely Ray, Kafka mocks, or DB connections).
3. Implement `pytest-xdist` or batching if necessary.
4. Ensure `pytest` runs to completion.

# Discussion/Comments

- 2026-02-06 Pickle Rick: Fixed by surgically disabling Ray initialization in `src/utils/distributed.py` during tests and enforcing aggressive mocks in `tests/mock_all.py`. Tests now run to completion (Exit Code 1) in <30s. OOM eliminated.
