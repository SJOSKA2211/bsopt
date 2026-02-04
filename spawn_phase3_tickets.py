import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "stream001": {
        "dir": "stream001",
        "path": "linear_ticket_stream001.md",
        "content": """---
id: stream001
title: Optimize Kafka Producer with msgspec Buffering
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, streaming]
assignee: Pickle Rick
---

# Description

## Problem to solve
Redundant serialization in MarketDataProducer increases latency.

## Solution
Use msgspec for internal data handling and optimize the batch production path.
"""
    },
    "aiops002": {
        "dir": "aiops002",
        "path": "linear_ticket_aiops002.md",
        "content": """---
id: aiops002
title: Implement Targeted Remediation in SelfHealingOrchestrator
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [reliability, aiops]
assignee: Pickle Rick
---

# Description

## Problem to solve
SelfHealingOrchestrator triggers every remediator for every anomaly, which is inefficient and risky.

## Solution
Implement a mapping between anomaly types and specific remediators to ensure targeted actions.
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
