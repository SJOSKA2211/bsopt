import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "math001": {
        "dir": "math001",
        "path": "linear_ticket_math001.md",
        "content": """---
id: math001
title: Implement Full Reiner-Rubinstein Barrier Model
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [math, pricing, exotics]
assignee: Pickle Rick
---

# Description

## Problem to solve
Analytical barrier option pricer uses placeholders and is missing critical mathematical components.

## Solution
Implement the full set of Reiner-Rubinstein (1991) formulas (A through F) to support all 8 standard barrier option types.
"""
    },
    "scrap002": {
        "dir": "scrap002",
        "path": "linear_ticket_scrap002.md",
        "content": """---
id: scrap002
title: Optimize Scraper Symbol Mapping with O(1) Lookup
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, scraper]
assignee: Pickle Rick
---

# Description

## Problem to solve
NSE Scraper uses sequential substring matching for symbol mapping, which is O(N) and inefficient.

## Solution
Implement a pre-computed exact-match cache and a more efficient keyword matching strategy.
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
