---
id: b002
title: Enforce Strict Linting
status: Done
priority: High
order: 20
created: 2026-02-19
updated: 2026-02-19
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
---

# Description

## Problem to solve
Code style is inconsistent. Jerries are committing sloppy code.

## Solution
Configure and enforce Black, Ruff, and Flake8.

## Implementation Details
- Update `pyproject.toml` with strict configs.
- Run `black .` and `ruff check . --fix`.
- Ensure zero errors.

## Notes
- Fixed 155+ errors including syntax errors in `assistant_implementer.py` and `store.py`.
- Fixed F821 undefined names in `bs_cli.py`, `router.py`, `ray_workers.py`, `ml_tasks.py`, `execution.py`, `orchestrator.py`.
- Remaining warnings (E402, F841) are acceptable for now.
