-- Migration: 003_add_session_columns.sql
-- Description: Adds missing columns to the sessions table required by BetterAuth.
-- These columns (ip_address, user_agent, updated_at) are expected by the
-- BetterAuthSession ORM model but were missing from the original schema.

BEGIN;

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS ip_address VARCHAR(50),
    ADD COLUMN IF NOT EXISTS user_agent TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

COMMIT;
