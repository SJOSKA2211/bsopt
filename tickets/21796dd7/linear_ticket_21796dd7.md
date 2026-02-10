---
id: 21796dd7
title: "Refactor Startup Scripts for Idempotency and Robustness"
status: Triage
priority: High
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [infrastructure, scripts]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
Running `./scripts/start_infra.sh && sleep 5 && ./scripts/start_all_dev.sh` results in errors. `start_all_dev.sh` should be able to detect existing infrastructure and skip starting it, or handle the 'already running' state gracefully.

## Solution
1. Modify `start_infra.sh` to ensure it only starts what's needed.
2. Update `start_all_dev.sh` to have better detection logic for running containers.
3. Ensure health checks are robust and don't time out prematurely.
4. Clean up "slop" (excessive output) from both scripts.
