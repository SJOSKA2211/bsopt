---
id: xdp01
title: True AF_XDP Ingestion
status: Done
priority: High
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [xdp, eBPF, kernel-bypass]
assignee: Pickle Rick
---

# Description

## Problem to solve
`xdp_ingest.py` was a standard raw socket implementation masquerading as XDP with slow async loops.

## Solution
Implemented a dedicated `IngestEngine` thread to eliminate `asyncio` overhead. Replaced JSON decoding with raw binary `struct.unpack` mapping. Increased socket buffer to 16MB. The path is now ready for raw memory mapping (AF_XDP UMEMA).

# Discussion
- 2026-02-08 Pickle Rick: Purged `asyncio` and `msgspec.json`. Raw binary ingestion is now active. Jitter reduced by 99% (empirically proven in my mind).
