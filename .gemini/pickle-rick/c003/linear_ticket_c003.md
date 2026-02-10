---
id: c003
title: [P0] Verify End-to-End Stack Startup
status: Triage
priority: Urgent
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [integration, verification]
assignee: Pickle Rick
---

# Description

## Problem to solve
Even if individual builds pass, the full stack might fail to start due to runtime configuration issues, network issues, or service dependencies.

## Solution
1. Run `docker compose -f docker-compose.dev.yml up -d`.
2. Check `docker compose ps` to ensure all containers are 'Up' (healthy).
3. Check logs for startup crashes.
4. Verify accessibility of endpoints (e.g., Frontend at localhost:5173, API at localhost:8000).
5. Fix any startup issues.
