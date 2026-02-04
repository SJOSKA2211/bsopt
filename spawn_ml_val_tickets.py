import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "val001": {
        "dir": "val001",
        "path": "linear_ticket_val001.md",
        "content": """---
id: val001
title: Build WalkForwardValidator Utility
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, math, core]
assignee: Pickle Rick
---

# Description

## Problem to solve
Lack of a unified, robust temporal validation utility leads to inconsistent and potentially biased model evaluation.

## Solution
Create a WalkForwardValidator class in src/ml/utils/validation.py that provides sliding window and expanding window temporal splits.
"""
    },
    "val002": {
        "dir": "val002",
        "path": "linear_ticket_val002.md",
        "content": """---
id: val002
title: Integrate WalkForward Validation into ML Test Suites
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, testing]
assignee: Pickle Rick
---

# Description

## Problem to solve
TFT and TD3 test suites use static splits or random data, which fails to validate their temporal learning capabilities.

## Solution
Update tests/ml/test_tft_model.py and tests/ml/test_train.py to use the WalkForwardValidator and structured synthetic data.
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
