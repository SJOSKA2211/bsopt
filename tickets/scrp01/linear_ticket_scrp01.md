---
id: scrp01
title: Sanitize Management Scripts
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [scripts, cleanup]
assignee: Morty
---

# Description

## Problem to solve
Scripts like `verify_god_mode.py`, `setup_pki.sh`, and `enforce_venv.py` are full of "Singularity" slop.

## Solution
1. Remove all rocket emojis and marketing text.
2. Rename "God Mode" to "Performance" or "Validation".
3. Ensure they are professional.
