---
id: neon001
title: "Migration: PostgreSQL to Neon"
status: Done
priority: Urgent
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: /home/kamau/bsopt/.gemini/pickle-rick/tickets/linear_ticket_parent.md
    title: Parent Ticket
labels: [migration, data]
assignee: Morty
---

# Description

## Problem to solve
The current PostgreSQL setup is static and hard to scale. We need the branching and serverless features of Neon to accelerate our quant development lifecycle.

## Solution
1. Export current schema from `init-scripts/`.
2. Configure Neon project and branching.
3. Update `src/shared/config.py` to use Neon connection strings.
4. Implement connection pooling (pgbouncer/Neon proxy) for the API and Workers.
