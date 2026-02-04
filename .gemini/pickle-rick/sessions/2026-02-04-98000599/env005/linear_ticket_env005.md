---
id: env005
title: Enforce .venv and Environment Standards
status: Done
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [devops, environment]
assignee: Pickle Rick
---

# Description

## Problem to solve
Environment consistency is key. We need to enforce `.venv` usage.

## Solution
Ensure all scripts and setup tools use/require `.venv`.

# Comments
- Created `scripts/enforce_venv.py`.
- Updated `bs_cli.py` to check for venv on startup.

