---
id: scrap02
title: Optimize Scraper Engine & Remove Slop
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [scraper, cleanup, performance]
assignee: Morty
---

# Description

## Problem to solve
`engine.py` has inline imports and marketing fluff.

## Solution
1. Move `orjson` and `pandas` imports to the top.
2. Remove "Singularity" and "SOTA" branding.
3. Clean up the main entry point loop.
