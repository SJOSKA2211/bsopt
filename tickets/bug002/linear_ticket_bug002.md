---
id: bug002
title: "Fix auth-service /api/auth/login 404 routing issue"
status: Triage
priority: Urgent
project: project
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../../linear_ticket_parent.md
    title: Parent Ticket
  - url: /home/kamau/bsopt/src/auth-service/plans/plan_f8e2a1b9_2026-02-03.md
    title: Research Context
labels: bug, auth, typescript, urgent
assignee: Pickle Rick
---

# Description

## Problem to solve
The `auth-service` (TypeScript/Hono) returns a 404 Not Found for POST requests to `/api/auth/login`. This blocks the entire authentication flow.

## Solution
Analyze `src/auth-service/src/index.ts` and `src/auth-service/src/auth.ts`. Ensure that the router is correctly configured to match and delegate auth requests to the `better-auth` handler. Correct any path prefix mismatches.
