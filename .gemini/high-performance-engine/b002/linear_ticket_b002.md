---
id: b002
title: [P0] Audit & Fix Node Services Builds (Auth, Frontend)
status: Done
priority: Urgent
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [frontend, node, docker]
assignee: High-Performance Engine
---

# Description

## Problem to solve
The Node.js-based services (auth-service, frontend) may have build errors (npm install failures, missing packages, build script errors).

## Solution
1. Run `docker compose -f docker-compose.dev.yml build auth-service frontend`.
2. Analyze build logs for errors.
3. Fix any identified errors in `docker/Dockerfile.*` or `package.json`.
4. Verify that these specific services build successfully.
