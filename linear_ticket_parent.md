---
id: parent
title: [Epic] Total Codebase Purification
status: Todo
priority: High
order: 0
created: 2026-02-20
updated: 2026-02-20
links: []
---

# Description

## Problem to solve
The `bsopt` codebase is in a critical state of disrepair with broken builds, missing dependencies, syntax errors, and zero verifiable test coverage.

## Solution
Execute a systematic purification of the codebase, fixing all errors, enabling the Dockerized test runner, and achieving >=99% coverage.

## Implementation Details
- Fix `src/api/main.py` syntax.
- Restore `src/pricing/quant_utils.py` exports.
- Fix all `ImportError`s in tests.
- Verify `make test-all` runs in Docker.
- Achieve >=99% coverage.
- Fix all linting errors.
