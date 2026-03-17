---
id: c001
title: Fix Syntax & Core Imports (The "It Won't Even Run" Ticket)
status: Done
priority: High
order: 10
created: 2026-02-20
updated: 2026-02-20
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
  - url: research_20260220.md
    title: Research Document
  - url: research_review.md
    title: Research Review
  - url: plan_20260220.md
    title: Implementation Plan
  - url: plan_review.md
    title: Plan Review
---

# Description

## Problem to solve
Test collection fails with 29 errors.
- `src/api/main.py` has a `SyntaxError` on line 130.
- `src/pricing/quant_utils.py` is missing exports `gpu_mc_european_price` and `scalar_bs_price_jit`.
- `tests/*` files cannot import the code they are supposed to test.

## Solution
Fix the syntax error and implement/stub the missing functions to allow test collection.

## Implementation Details
1.  **Fix Syntax**: Correct `src/api/main.py`.
2.  **Fix Imports**: Add the missing functions to `src/pricing/quant_utils.py`. Since `gpu_mc_european_price` implies CUDA, create a CPU fallback or stub for the test environment.
3.  **Verify Collection**: Run `pytest --collect-only` to ensure all tests can be discovered.
