---
id: c000_fix
title: "Fix Indentation in Pricing Service"
status: Done
priority: Urgent
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, core]
assignee: High-Performance Engine
---

# Description

## Problem to solve
`src/services/pricing_service.py` has an `IndentationError` at line 118. This blocks all tests from running.

## Solution
Fix the indentation of `calculate_greeks` and `clear_cache` methods in `src/services/pricing_service.py`.
