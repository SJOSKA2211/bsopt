---
id: 6a2b1c8f
title: "Fix Environment and Baseline Coverage"
status: Plan in Review
priority: Urgent
project: project
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
  - url: ./research_20260204.md
    title: Research Document
  - url: ./plan_20260204.md
    title: Implementation Plan
labels: [core, testing, infrastructure]
assignee: Pickle Rick
---

# Description

## Problem to solve
Tests are currently failing to run because of missing environment variables (`REDIS_URL`, `JWT_SECRET`). This prevents us from getting an accurate baseline.

## Solution
Configure the test environment with necessary mocks or dummy values to allow `pytest` to run fully. Run all existing tests and generate a comprehensive `coverage.xml`.

# Discussion/Comments
