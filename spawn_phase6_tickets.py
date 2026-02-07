import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "test001": {
        "dir": "test001",
        "path": "linear_ticket_test001.md",
        "content": """---
id: test001
title: Fix Broken Auth and Rate Limit Tests
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, testing, security]
assignee: Pickle Rick
---

# Description

## Problem to solve
Core auth and rate limiting tests are failing due to architectural changes.

## Solution
Update tests/api/routes/test_auth_routes.py and tests/functional/test_api_v1.py to reflect the new OIDC triad and rate limit tiers.
""",
    },
    "test002": {
        "dir": "test002",
        "path": "linear_ticket_test002.md",
        "content": """---
id: test002
title: Implement Walk-Forward Validation for ML Tests
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, math, testing]
assignee: Pickle Rick
---

# Description

## Problem to solve
Existing ML tests use random shuffling, which is invalid for time-series data.

## Solution
Implement a Walk-Forward validation utility and apply it to the TFT and TD3 test suites.
""",
    },
    "test003": {
        "dir": "test003",
        "path": "linear_ticket_test003.md",
        "content": """---
id: test003
title: Boost Coverage for SIMD and OAuth Logic
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, security, testing]
assignee: Pickle Rick
---

# Description

## Problem to solve
New performance-critical and security-critical logic lacks sufficient test coverage.

## Solution
Add unit tests for WASM SIMD greeks and integration tests for the OIDC discovery/JWKS flow.
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
