import os

session_root = "/home/kamau/.gemini/extensions/pickle-rick/sessions/2026-02-04-76d95b11"

tickets = {
    "trade001": {
        "dir": "trade001",
        "path": "linear_ticket_trade001.md",
        "content": """---
id: trade001
title: Bridge OrderExecutor to DeFiOptionsProtocol
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [trading, blockchain]
assignee: Pickle Rick
---

# Description

## Problem to solve
OrderExecutor uses mock hashes instead of real blockchain transactions.

## Solution
Update OrderExecutor.execute_order to call DeFiOptionsProtocol.buy_option.
""",
    },
    "back002": {
        "dir": "back002",
        "path": "linear_ticket_back002.md",
        "content": """---
id: back002
title: Parallelize BacktestEngine using Ray
status: Triage
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [performance, backtesting]
assignee: Pickle Rick
---

# Description

## Problem to solve
BacktestEngine runs sequentially, limiting HPO and multi-strategy runs.

## Solution
Integrate Ray to parallelize the execution of multiple backtest scenarios.
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
