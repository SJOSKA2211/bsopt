---
id: scrap01
title: Optimize Stealth Scrapers
status: Backlog
priority: Medium
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [scraper, stealth, optimization]
assignee: Morty
---

# Description

## Problem to solve
`stealth.py` is not stealthy. It has emojis in the comments and "Singularity" branding.

## Solution
1. Remove all emojis and marketing fluff.
2. Ensure the GET request logic is actually optimized for high-throughput without being detected.
