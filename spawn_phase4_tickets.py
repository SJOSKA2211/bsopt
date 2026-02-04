import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "gate001": {
        "dir": "gate001",
        "path": "linear_ticket_gate001.md",
        "content": """---
id: gate001
title: Optimize Gateway Worker Startup with Shared SDL
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, gateway]
assignee: Pickle Rick
---

# Description

## Problem to solve
Every Gateway worker independently introspects subgraphs, causing redundant network traffic and slow startup.

## Solution
Update the Primary process to perform introspection once and share the SDL with workers via an environment variable or shared file.
"""
    },
    "wasm002": {
        "dir": "wasm002",
        "path": "linear_ticket_wasm002.md",
        "content": """---
id: wasm002
title: Refactor WASM SIMD Path for Safety and Performance
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, wasm, rust]
assignee: Pickle Rick
---

# Description

## Problem to solve
WASM SIMD path uses unsafe memory transmutations and only parallelizes the Price calculation.

## Solution
Refactor batch_calculate_simd to include SIMD-accelerated Greeks and use safer memory access patterns.
"""
    }
}

for key, ticket in tickets.items():
    dir_path = os.path.join(session_root, ticket["dir"])
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, ticket["path"])
    
    with open(file_path, "w") as f:
        f.write(ticket["content"])
    print(f"Created: {file_path}")
