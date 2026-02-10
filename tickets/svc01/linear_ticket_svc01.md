---
id: svc01
title: Optimize Services & Remove Marketing Fluff
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [services, optimization, cleanup]
assignee: The User
---

# Description

## Problem to solve
`pricing_service.py` and `ml_service.py` have "Object Creation Slop" (inline imports) and "Optimized" branding.

## Solution
1. Move all imports to the top of the file.
2. Remove "Optimized Refactored" and related marketing headers.
3. Clean up the SHM overloading logic in `ml_service.py` to be more explicit (e.g., use a dedicated metadata field if gRPC allows, or at least a constant flag).
