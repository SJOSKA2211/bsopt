---
id: front01
title: Optimize Frontend Pricing Worker
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [frontend, wasm, optimization]
assignee: Morty
---

# Description

## Problem to solve
`pricing.worker.ts` has "Optimized" slop and inefficient buffer handling.

## Solution
1. Remove all marketing headers and emojis.
2. Ensure `postMessage` uses the transferable array buffers properly.
3. Optimize the switch-case logic.
