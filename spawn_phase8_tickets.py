import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "task001": {
        "dir": "task001",
        "path": "linear_ticket_task001.md",
        "content": """---
id: task001
title: Optimize Task Initialization with Engine Caching
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, task]
assignee: Pickle Rick
---

# Description

## Problem to solve
Pricing tasks re-initialize strategies on every run, adding unnecessary overhead.

## Solution
Implement a worker-local cache for engine instances in src/tasks/pricing_tasks.py.
""",
    },
    "task002": {
        "dir": "task002",
        "path": "linear_ticket_task002.md",
        "content": """---
id: task002
title: Optimize Batch Task Serialization
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, task]
assignee: Pickle Rick
---

# Description

## Problem to solve
Batch pricing tasks use slow manual loops for result formatting.

## Solution
Use msgspec or optimized dictionary creation to speed up batch result serialization.
""",
    },
}

for key, ticket in tickets.items():
    dir_path = os.path.join(session_root, ticket["dir"])
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, ticket["path"])

    with open(file_path, "w") as f:
        f.write(ticket["content"])
    print(f"Created: {file_path}")
