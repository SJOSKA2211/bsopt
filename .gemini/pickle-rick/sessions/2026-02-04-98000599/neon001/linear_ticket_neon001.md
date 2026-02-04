---
id: neon001
title: Migrate Backend to Neon (Postgres)
status: Done
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [backend, database, neon]
assignee: Pickle Rick
---

# Description

## Problem to solve
The current data persistence layer needs to be scalable and serverless.

## Solution
Migrate the backend to use Neon (Serverless Postgres). Configure connection strings, migrate schemas, and ensure the API uses the new DB.

# Comments
- Updated `src/ml/autonomous_pipeline.py` to use `settings.DATABASE_URL`.
- Created `NEON_SETUP.md` with instructions.
- Upgraded `src/database/verify.py` to use SQLAlchemy.
- Verified schema compatibility.

