---
id: rl01
title: RL Policy Advancement
status: Done
priority: High
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [rl, deep-learning, sota]
assignee: Pickle Rick
---

# Description

## Problem to solve
`offline_train.py` is a skeleton. GNN is basic.

## Solution
1. Implement Return-to-go computation and trajectory loading in `offline_train.py`.
2. Upgrade `gnn_policy.py` to use Graph Attention Layers (GAT) or Transformer-based graph layers.
3. Ensure end-to-end training functionality for the Decision Transformer.

# Discussion
- 2026-02-08 Pickle Rick: Implemented full offline training loop for Decision Transformer. Upgraded GNN Feature Extractor to use GATConv (Graph Attention) for superior option surface modeling. Purged Jerry-level fallbacks.
