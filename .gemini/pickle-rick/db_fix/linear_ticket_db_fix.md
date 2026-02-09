---
id: db_fix
title: "Fix Postgres Schema and Missing Functions"
status: Done
priority: Urgent
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [infra, postgres, bug]
assignee: Morty
---

# Description

## Problem to solve
Postgres is throwing errors: `function refresh_market_stats() does not exist` and `column "delta" does not exist`.

## Solution
1. Locate DB init scripts.
2. Add missing `refresh_market_stats()` function.
3. Add missing columns `delta`, `gamma`, `implied_volatility` to `options_prices`.
