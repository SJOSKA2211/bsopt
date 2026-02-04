---
id: auth001
title: Fix Auth Route Failures
status: Triage
priority: Urgent
project: BS-OPT
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, auth, critical]
assignee: Pickle Rick
---

# Description

## Problem to solve
`tests/api/routes/test_auth_routes.py` is failing. Users cannot register or verify email.

## Solution
Debug and fix the auth flow. Verify with tests.
