---
id: data01
title: Audit & Optimize Data Ingestion
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [data, performance]
assignee: Morty
---

# Description

## Problem to solve
`xdp_ingest.py` might be using inefficient parsing. We need to ensure it's high-throughput.

## Solution
1. Read `xdp_ingest.py`.
2. Look for serialization slop.
3. Optimize the ingest loop.
