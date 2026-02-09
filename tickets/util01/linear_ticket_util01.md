---
id: util01
title: Clean Up Distributed & SHM Utilities
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [utils, cleanup, performance]
assignee: Morty
---

# Description

## Problem to solve
`distributed.py` and `shm_context.py` contain misleading marketing fluff ("Optimized", "ADVANCED") and stubbed features that don't work.

## Solution
1. Remove all branding slop and rocket emojis.
2. Replace stubbed `IOUringPersister` logic with a standard, documented synchronous path if `liburing` is not actually being used for anything meaningful yet.
3. Clean up docstrings to reflect reality.
