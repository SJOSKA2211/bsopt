---
id: wasm01
title: Audit Rust/WASM Core
status: Backlog
priority: Medium
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [rust, wasm, performance]
assignee: Morty
---

# Description

## Problem to solve
We have a Rust/WASM core but don't know if it's actually "God Mode" or just more slop.

## Solution
1. Read `src/wasm/src/lib.rs`.
2. Ensure SIMD is actually being used where claimed.
3. Remove branding fluff.
