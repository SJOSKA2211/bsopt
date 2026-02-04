---
id: ml002
title: "Advanced Forecasting: Transforming the Forecasting Pipeline"
status: Done
priority: Medium
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
The forecasting models in src/ml/forecasting likely use basic LSTM/RNN architectures which are inferior to modern temporal attention mechanisms.

## Solution
1. Audit src/ml/forecasting and src/ml/architectures.
2. Implement a Temporal Fusion Transformer (TFT) or similar Attention-based model for price and volatility forecasting.
3. Fine-tune hyperparameters for high-frequency volatility prediction.
