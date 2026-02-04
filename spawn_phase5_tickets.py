import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "ml001": {
        "dir": "ml001",
        "path": "linear_ticket_ml001.md",
        "content": """---
id: ml001
title: Implement Temporal Validation in ML Pipeline
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, math, security]
assignee: Pickle Rick
---

# Description

## Problem to solve
Random shuffling in train_test_split causes future-to-past data leakage in time-series models.

## Solution
Replace random shuffling with TimeSeriesSplit or sequential index slicing to ensure valid temporal validation.
"""
    },
    "ml002": {
        "dir": "ml002",
        "path": "linear_ticket_ml002.md",
        "content": """---
id: ml002
title: Unify MLflow Tracking with Neon Backend
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, backend]
assignee: Pickle Rick
---

# Description

## Problem to solve
MLflow logs to local files, which is not scalable or shareable across the platform.

## Solution
Update the MLflow tracking URI to use the Neon database URL (postgres backend).
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
