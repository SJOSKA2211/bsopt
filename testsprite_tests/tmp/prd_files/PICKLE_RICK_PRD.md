# PICKLE RICK PRD: Fix Startup & Lint Slop

## Problem
1.  **Lint Failure**: `pyproject.toml` contains invalid TOML syntax for `ruff.lint.per-file-ignores`. It uses a nested list instead of a map, causing `ruff` to crash immediately.
2.  **Startup Redundancy**: `start_all_dev.sh` duplicates infrastructure startup logic found in `start_infra.sh`. When run sequentially, this causes redundant checks and "slop" code execution.

## Goal
1.  Fix `pyproject.toml` so `ruff check .` runs successfully.
2.  Refactor `start_all_dev.sh` to remove redundant infrastructure logic and properly delegate to `start_infra.sh` or handle existing services gracefully.
3.  Ensure the command `bash ./scripts/start_infra.sh && sleep 5 && bash ./scripts/start_all_dev.sh` works flawlessly.

## Technical Approach
1.  **Fix Config**: Rewrite the `[tool.ruff.lint]` section in `pyproject.toml` to use correct dictionary syntax for `per-file-ignores`.
2.  **Refactor Script**:
    -   In `start_all_dev.sh`, replace the redundant `docker compose up` block with a check.
    -   If infra is missing, call `./scripts/start_infra.sh` instead of duplicating the logic.

## Verification
1.  `ruff check .` executes without a TOML parse error.
2.  The sequential startup command runs without errors.
