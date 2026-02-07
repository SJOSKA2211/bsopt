---
id: rl01
title: Optimize RL Transformer Policy & Extractor
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [rl, pytorch, optimization]
assignee: Morty
---

# Description

## Problem to solve
The `TransformerSingularityExtractor` and `DecisionTransformer` in `transformer_policy.py` use inefficient tensor operations (manual interleaving, unsqueeze/squeeze in tight loops) and have "Singularity" slop.

## Solution
1. Rename `TransformerSingularityExtractor` to `TransformerFeatureExtractor`.
2. Optimize tensor interleaving in `DecisionTransformer.forward` using `torch.stack` and `reshape` instead of `zeros` + slice assignment.
3. Remove "Singularity" and "SOTA" comments.
