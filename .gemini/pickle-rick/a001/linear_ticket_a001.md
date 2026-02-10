---
id: a001
title: [P0] Audit & Fix Core Backend Builds (API, Pricing, Worker)
status: Done
priority: Urgent
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [backend, python, docker]
assignee: Pickle Rick
---

# Description

## Problem to solve
The Python-based backend services (api, neural-pricing, worker-ml, scraper) may have build errors in their Dockerfiles or dependencies.

## Solution
1. Run `docker compose -f docker-compose.dev.yml build api neural-pricing worker-ml scraper`.
2. Analyze build logs for errors (missing dependencies, syntax errors, permission issues).
3. Fix any identified errors in `docker/Dockerfile.*` or `requirements*.txt`.
4. Verify that these specific services build successfully.
