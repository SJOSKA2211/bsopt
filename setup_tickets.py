import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "parent": {
        "path": "linear_ticket_parent.md",
        "content": """---
id: parent
title: [Epic] BSOpt Singularity Upgrade
status: Backlog
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: prd.md
    title: PRD
labels: [epic, overhaul]
assignee: Pickle Rick
---

# Description

## Problem to solve
The BSOpt platform requires a comprehensive overhaul to achieve "Singularity" status, including backend migration, security standardization, and ML model evolution.

## Solution
Execute the atomic child tickets covering Neon, OAuth, Optimization, and Transformer RL.
"""
    },
    "neon001": {
        "dir": "neon001",
        "path": "linear_ticket_neon001.md",
        "content": """---
id: neon001
title: Migrate Backend to Neon (Serverless Postgres)
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [backend, database, neon]
assignee: Pickle Rick
---

# Description

## Problem to solve
Local/Standard DB setup lacks serverless scalability.

## Solution
Configure Neon connection and migrate existing schemas.
"""
    },
    "oauth002": {
        "dir": "oauth002",
        "path": "linear_ticket_oauth002.md",
        "content": """---
id: oauth002
title: Implement OAuth 2.0 Triad (Auth/Client/Resource)
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [security, oauth]
assignee: Pickle Rick
---

# Description

## Problem to solve
System lacks secure OIDC-compliant authentication.

## Solution
Implement Auth-Server, Client-App, and Resource-Server logic.
"""
    },
    "opt003": {
        "dir": "opt003",
        "path": "linear_ticket_opt003.md",
        "content": """---
id: opt003
title: Codebase Audit & Extreme Optimization
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [optimization, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
Codebase has unoptimized functions and technical debt.

## Solution
Audit all modules and implement performance improvements.
"""
    },
    "rl004": {
        "dir": "rl004",
        "path": "linear_ticket_rl004.md",
        "content": """---
id: rl004
title: Integrate Transformer Models into RL Pipeline
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, rl, transformer]
assignee: Pickle Rick
---

# Description

## Problem to solve
Current RL agents lack temporal attention.

## Solution
Replace/augment policy networks with Transformer architectures.
"""
    },
    "cmt005": {
        "dir": "cmt005",
        "path": "linear_ticket_cmt005.md",
        "content": """---
id: cmt005
title: Refine and Optimize Codebase Comments
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [documentation, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
Outdated or redundant comments clutter the code.

## Solution
Fine-tune all docstrings and comments for clarity and precision.
"""
    },
    "venv006": {
        "dir": "venv006",
        "path": "linear_ticket_venv006.md",
        "content": """---
id: venv006
title: Enforce .venv Usage Across Workspace
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [devops, environment]
assignee: Pickle Rick
---

# Description

## Problem to solve
Environment inconsistency risks Jerry-work bugs.

## Solution
Add checks to ensure script execution only within the .venv.
"""
    }
}

os.makedirs(session_root, exist_ok=True)

for key, ticket in tickets.items():
    if "dir" in ticket:
        dir_path = os.path.join(session_root, ticket["dir"])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, ticket["path"])
    else:
        file_path = os.path.join(session_root, ticket["path"])
    
    with open(file_path, "w") as f:
        f.write(ticket["content"])
    print(f"Created: {file_path}")
