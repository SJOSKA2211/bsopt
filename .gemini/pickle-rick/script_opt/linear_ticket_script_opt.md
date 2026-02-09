---
id: script_opt
title: "Optimize Startup Scripts Logging and Error Handling"
status: Done
priority: Medium
project: project
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [devops, shell]
assignee: Morty
---

# Description

## Problem to solve
Startup scripts are a bit "quiet" and don't bail properly on certain infrastructure failures.

## Solution
1. Enhance `start_all_dev.sh` with better health check reporting.
2. Ensure `start_infra.sh` output is piped correctly.
