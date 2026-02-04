---
id: 7d3e5f9a
title: "Unit Test Audit & Mapping"
status: Triage
priority: High
project: project
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [core, testing, research]
assignee: Pickle Rick
---

# Description

## Problem to solve
We need to know exactly which files in `src/` are dragging down the coverage.

## Solution
Analyze the `coverage.xml` baseline and map out the biggest "holes". Prioritize files by their importance to the core manifold (e.g. `src/pricing/`, `src/aiops/`).

# Discussion/Comments
