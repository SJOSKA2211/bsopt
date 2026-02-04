---
id: epic001
title: "[Epic] Codebase Optimization & Unified Auth (Neon Integration)"
status: "Backlog"
priority: Urgent
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: /home/kamau/bsopt/PICKLE_RICK_PRD.md
    title: PRD
labels: [epic, core, auth, ml, data]
assignee: Pickle Rick
---

# Description

## Problem to solve
The bsopt codebase is currently a fragmented mess of unoptimized pricing logic, insecure/scattered auth code, and legacy database infrastructure. It lacks temporal awareness in RL and has significant latency overhead.

## Solution
Perform a full-scale refactor to implement a unified OAuth2 stack, migrate to Neon serverless Postgres, integrate Transformer policies into the RL loop, and optimize all core math functions using hardware-aware vectorization.
