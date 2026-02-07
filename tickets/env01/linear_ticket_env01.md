---
id: env01
title: Audit & Optimize Trading Environment
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [env, gym, optimization]
assignee: Morty
---

# Description

## Problem to solve
`trading_env.py` might be slow. RL training is limited by the environment's `step` speed.

## Solution
1. Read `trading_env.py`.
2. Look for vectorization opportunities.
3. Optimize reward calculation.
