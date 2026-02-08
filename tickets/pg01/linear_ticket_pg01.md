---
id: pg01
title: Postgres Purity Enforcement
status: Done
priority: Urgent
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [database, infrastructure, purity]
assignee: Pickle Rick
---

# Description

## Problem to solve
Lingering "Neon" references and dependencies.

## Solution
1. Grep and destroy all occurrences of "neon" in comments, configs, and documentation.
2. Remove cloud-specific setup files.
3. Ensure `src/config.py` uses standard `postgresql://` without any cloud-specific branding.

# Discussion
- 2026-02-08 Pickle Rick: Deleted `NEON_SETUP.md` and `99-neon-migration.sql`. Scrubbed `config.py` comments. The timeline is now 100% Native Postgres.
