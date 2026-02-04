---
id: rl001
title: "Transformer-based RL Feature Extractor"
status: Triage
priority: High
project: bsopt
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, rl]
assignee: Pickle Rick
---

# Description
## Problem to solve
RL agents use basic MLP policies and miss temporal market context.

## Solution
Implement an attention-based feature extractor for state representation in the RL pipeline.
