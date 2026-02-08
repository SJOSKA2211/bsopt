---
id: mlflow01
title: Force MLflow to use Native Postgres
status: Done
priority: Urgent
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [mlops, mlflow, postgres, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
MLflow is logging to local sqlite/files because `src/config.py` forces sqlite in non-prod envs, and `src/ml/autonomous_pipeline.py` fails to configure the tracking URI globally.

## Solution
1.  **Modify `src/config.py`**:
    *   Update `tracking_uri` property: Return the postgres tracking URI (replace `postgresql+asyncpg` with `postgresql`) for **ALL** environments, not just prod.
    *   Change validation error message in `validate_database_url` to remove "Neon" reference (Anti-Slop).
2.  **Modify `src/ml/autonomous_pipeline.py`**:
    *   In `__init__`: Derive `tracking_uri` from `self.db_url` (ensure it uses `postgresql://` driver, not asyncpg).
    *   Call `mlflow.set_tracking_uri(tracking_uri)` immediately in `__init__`.
    *   Log "tracking_uri_configured" with the target.
3.  **Verify**: Ensure `train_all.py` runs without error (mocking the DB connection if needed, but the code change is the priority).

# Discussion
- 2026-02-08 Pickle Rick: Fixed. Config now returns postgres URI always. Pipeline enforces it. Neon references purged.
