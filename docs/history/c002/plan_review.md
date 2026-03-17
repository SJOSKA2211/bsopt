# Plan Review: Docker & Test Runner Stabilization (Ticket c002)

**Status**:  APPROVED
**Reviewed**: 2026-02-20

## 1. Structural Integrity
- [x] **Atomic Phases**: Phasing is correct. Database initialization MUST happen before the tests try to connect.
- [x] **Worktree Safe**: The plan modifies configuration files and tests, suitable for a localized implementation.

*Architect Comments*: Logical progression from infrastructure (SQL) to test config to global config.

## 2. Specificity & Clarity
- [x] **File-Level Detail**: Cites `init-scripts/00-core-schema.sql`, `tests/conftest.py`, `src/api/websockets/manager.py`, and `src/tasks/celery_app.py`.
- [x] **No "Magic"**: Steps are explicit about what to check and what to change.

*Architect Comments*: The use of `INSIDE_DOCKER` as a guard for fallbacks is a solid pattern.

## 3. Verification & Safety
- [x] **Automated Tests**: Uses `grep` and `py_compile` for verification.
- [x] **Manual Steps**: Verification steps are clear and reproducible.
- [x] **Rollback/Safety**: Database change is additive (`CREATE DATABASE`).

*Reviewer Comments*: Since `docker` isn't available in the sandbox, static verification (`py_compile` and `grep`) is the only viable path. The plan acknowledges this.

## 4. Architectural Risks
- Low risk. These are standard containerization best practices.

## 5. Recommendations
- Proceed to implementation. Ensure `tests/conftest.py` doesn't break local "host-mode" execution by checking for the environment variables first.
