---
id: rl001
title: "Transformer-based RL Feature Extractor"
status: Done
priority: High
project: bsopt
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, rl, transformer]
assignee: Pickle Rick
---

# Description
## Problem to solve
RL agent uses flat observation vectors, missing temporal context.

## Solution
Implement a Transformer-based feature extractor and refactor TradingEnvironment to provide sequence-based observations (window of 16).
