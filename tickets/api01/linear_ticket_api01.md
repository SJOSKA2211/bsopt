---
id: api01
title: Clean API Middleware & Responses
status: Backlog
priority: Medium
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [api, cleanup]
assignee: Morty
---

# Description

## Problem to solve
`api/responses.py` and `middleware/idempotency.py` have "Singularity" slop.

## Solution
Remove branding fluff and emojis.
