# Neon Database Migration PRD

## HR Eng

| Neon Database Migration PRD |  | Summary: Migrating the `bsopt` database from local/Docker-based PostgreSQL to Neon serverless PostgreSQL to enable better scaling, branching, and reliability. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: [User] **Intended audience**: Engineering | **Status**: Draft **Created**: February 3, 2026 | **Self Link**: [Link] **Context**: [Link] |

## Introduction
The `bsopt` platform requires a robust, scalable database. Neon offers serverless PostgreSQL that integrates well with our existing async architecture.

## Problem Statement
**Current Process:** Local PostgreSQL in Docker or manual setup.
**Pain Points:** 
- Manual scaling.
- No database branching for feature testing.
- Persistence risks on local dev machines.
**Importance:** Vital for moving towards a production-ready cloud-native architecture.

## Objective & Scope
**Objective:** Fully transition database operations to Neon.
**In-scope:**
- Configuration of `DATABASE_URL`.
- Installation of `asyncpg`.
- Schema migration/initialization.
- Connectivity verification.

## Product Requirements
1. **[Config Management]**: Create/Update `.env` with Neon connection string.
2. **[Dependency Management]**: Add `asyncpg` to requirements and install.
3. **[Schema Initialization]**: Apply `init-scripts/` to the new Neon instance.
4. **[Verification]**: Run `neon_health_check` to ensure 100% success.

## Assumptions
- The provided connection string is valid and has sufficient permissions.
- Neon instance supports the required extensions (Timescale/pgvector) if they are used in `init-scripts/`.
