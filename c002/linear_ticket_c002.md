---
id: c002
title: Docker & Test Runner Stabilization (The "It Works On My Machine" Ticket)
status: Done
priority: High
order: 20
created: 2026-02-20
updated: 2026-02-20
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
  - url: research_20260220.md
    title: Research Document
  - url: research_review.md
    title: Research Review
  - url: plan_20260220.md
    title: Implementation Plan
  - url: plan_review.md
    title: Plan Review
---

# Description

## Problem to solve
Tests must run in the containerized `test-runner` environment. The `docker-compose.yml` configures a `test-runner` service, but we need to ensure it has all dependencies and environment variables correctly set to run the suite without manual intervention.

## Solution
Make `make test-all` work flawlessly inside Docker.

## Implementation Details
1.  **Test Runner Config**: Check `docker/Dockerfile.ci` and `docker-compose.yml` for missing dependencies.
2.  **Environment Variables**: Ensure `.env.test` or environment injection works for `POSTGRES_DB`, `REDIS_URL`, etc.
3.  **Command Execution**: Run `make test-all` and fix any infrastructure/connection errors.
4.  **Mocking External Services**: If the tests try to hit real APIs or hardware, mock them.
