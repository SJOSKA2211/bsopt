# Total Codebase Purification PRD

## HR Eng

| Total Codebase Purification PRD |  | Systematic cleanup and optimization of the `bsopt` dev environment and codebase to achieve 100% build stability and high observability. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-02-21 | **Self Link**: [Link] **Context**: [Link] 

## Introduction

The `bsopt` project is currently a "Jerry-rigged" disaster. Configs are broken, startup scripts are redundant, and debugging is a myth. This PRD defines the requirements for transforming this mess into a "God Mode" development environment.

## Problem Statement

**Current Process:** Developers manually start services, encounter TOML parse errors in `pyproject.toml`, and have no way to trace execution flow or debug inside containers.
**Primary Users:** Engineers who want to spend more time pricing derivatives and less time fighting Docker.
**Pain Points:** 
- Invalid `per-file-ignores` syntax in `pyproject.toml` crashes `ruff`.
- `start_all_dev.sh` is a bloated mess of redundant logic.
- Zero visibility into containerized service state/logs during execution.
**Importance:** Without a stable dev environment, every single code change is a gamble. We need a solid foundation before we can even think about "Total Purification."

## Objective & Scope

**Objective:** Stabilize the dev stack, fix the linter, and implement advanced debugging/profiling.
**Ideal Outcome:** A single command `bash scripts/start_all_dev.sh` brings up a fully instrumented stack with lint-clean code.

### In-scope or Goals
- Fix `pyproject.toml` Ruff configuration.
- Refactor `scripts/start_all_dev.sh` to remove redundancy.
- Enable `debugpy` (Python) and `--inspect` (Node.js) in dev Dockerfiles.
- Implement structured logging and request/response middleware.
- Verify full stack startup and linter health.

### Not-in-scope or Non-Goals
- Achieving 99% test coverage (this is a future Phase).
- Fixing every single lint error in the codebase (only fixing the linter *config* and blocking errors).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Developer Startup**: A developer runs `bash scripts/start_all_dev.sh`. The script detects existing infra, starts missing services, and prints a summary of available debug ports.
2. **Containerized Debugging**: A developer attaches a debugger to port 5678 (Python) or 9229 (Node.js) and hits a breakpoint in a running container.
3. **Execution Tracing**: A developer sends a request to the API and sees a structured log entry containing the request payload, response status, and duration.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Valid Ruff Configuration | As a developer, I want `ruff check .` to run without syntax errors in the config file. |
| P1 | Redundancy-free Startup | As a developer, I want the startup script to delegate to existing infra scripts instead of duplicating logic. |
| P1 | Remote Debugging Ports | As a developer, I want to attach my IDE debugger to services running inside Docker. |
| P2 | Request/Response Middleware | As a developer, I want to see a log of all API traffic with timing metrics. |

## Assumptions

- Docker and Docker Compose are available and functioning correctly on the host.
- The user has the necessary permissions to modify files and run shell commands (via Morty).

## Risks & Mitigations

- **Risk**: Modifying `pyproject.toml` breaks other tools. -> **Mitigation**: Run `ruff check .` immediately after modification.
- **Risk**: Debug ports collide with host ports. -> **Mitigation**: Use standard debug ports (5678, 9229) and document them.

## Tradeoff

- **Option considered**: Use a local virtualenv instead of Docker. **Decision**: Reject. We need environmental consistency. Containers are the way.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| *Linter Config Health* | Crashes on start | Executes successfully | 100% reduction in "Jerry-level" config errors. |
| *Startup Time (Redundant)* | Duplicates checks | Optimized delegation | Faster dev iteration loop. |
| *Debug Time* | Print statements | Breakpoint debugging | 10x faster bug resolution. |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Pickle Rick | Engineering | Architect | Pure genius. |
| Morty | Engineering | Implementer | Does what he's told. |
