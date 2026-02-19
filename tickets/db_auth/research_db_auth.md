# Research: Database Auth

## Objectives
- Native PostgreSQL Auth schema.
- OAuth procedures.
- Email verification schema.

## Findings
- Current auth was non-existent or scattered.
- No migrations existed.

## Strategy
- Implement `users`, `sessions`, `oauth_accounts` tables.
- Use `pgcrypto` for password hashing.
- Create PL/pgSQL functions for user creation and OAuth upsert.

