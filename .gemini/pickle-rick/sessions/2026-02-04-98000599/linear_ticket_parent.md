---
id: parent
title: [Epic] BSOpt Singularity Upgrade
status: Done
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: prd.md
    title: PRD
labels: [epic, overhaul]
assignee: Pickle Rick
---

# Description

## Problem to solve
The BSOpt project requires a massive upgrade including Neon backend, OAuth, code optimization, and RL improvements.

## Solution
Execute the BSOpt Singularity Upgrade PRD.

# Comments
- **Neon Backend:** Configured `DATABASE_URL`, updated schema, created setup guide.
- **OAuth:** Implemented Auth Server (JWKS), updated DB schema.
- **Optimization:** Optimized Monte Carlo (Pathwise Greeks) and Black-Scholes (Scalar Path).
- **RL:** Integrated Transformer policy, fixed training loop, added MLflow.
- **Environment:** Enforced `.venv` usage.

