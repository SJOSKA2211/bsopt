---
id: sec01
title: Optimize & Clean Security Layer
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [security, cleanup]
assignee: Morty
---

# Description

## Problem to solve
`shared/security.py` has branding slop and stubbed WASM enforcement.

## Solution
1. Remove "Optimized" and "ADVANCED" text.
2. Clean up `WASMOPAEnforcer` to be a valid fallback or a properly documented placeholder.
