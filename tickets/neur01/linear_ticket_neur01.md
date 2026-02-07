---
id: neur01
title: Integrate Neural Engine into Production Path
status: Backlog
priority: Medium
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, neural, integration]
assignee: Morty
---

# Description

## Problem to solve
The `NeuralPricingEngine` is a skeleton and not used by the `PricingService`.

## Solution
1. Ensure `PricingEngineFactory` can load the Neural Engine.
2. Add a training task to Celery that updates the Neural Engine weights using JIT data.
3. Integrate the Neural Engine as an optional model in `PricingService.price_option`.
