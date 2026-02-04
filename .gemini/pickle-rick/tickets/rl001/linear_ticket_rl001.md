---
id: rl001
title: "Upgrade: TD3 RL Trainer with Transformer Policy"
status: Done
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: /home/kamau/bsopt/.gemini/pickle-rick/tickets/linear_ticket_parent.md
    title: Parent Ticket
labels: [upgrade, ml]
assignee: Morty
---

# Description

## Problem to solve
The current TD3 trading agent uses a standard MLP for state representation, which fails to capture long-term temporal dependencies in market data.

## Solution
1. Integrate src/ml/reinforcement_learning/transformer_policy.py into the TD3 trainer.
2. Update the environment observation space to include historical windows.
3. Fine-tune the Attention mechanism for state encoding.
4. Verify convergence on historical market data.
