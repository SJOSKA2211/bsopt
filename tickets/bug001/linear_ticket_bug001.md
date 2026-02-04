---
id: bug001
title: "Fix TradingEnvironment asset purchase cost bug"
status: Triage
priority: Urgent
project: project
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../../linear_ticket_parent.md
    title: Parent Ticket
  - url: /home/kamau/bsopt/tests/ml/test_trading_env_advanced.py
    title: Failing Test Reference
labels: bug, ml, rl, urgent
assignee: Pickle Rick
---

# Description

## Problem to solve
In `src/ml/reinforcement_learning/trading_env.py`, the `step` function correctly deducts transaction costs but FAILS to deduct the actual purchase price of assets from the cash balance. This allows the agent to "buy" assets for free, minus a small commission.

## Solution
Modify the `step` function to subtract `np.sum(trades * current_prices)` from `self.balance` in addition to the transaction costs. Ensure that selling assets (negative trades) correctly increases the balance.
