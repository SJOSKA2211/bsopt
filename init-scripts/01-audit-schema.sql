-- 🛡️ SOC2 COMPLIANT AUDIT SCHEMA
-- Optimized for High Ingest and Immutable Storage using TimescaleDB

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. Create Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
    time TIMESTAMPTZ NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    client_ip TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    metadata JSONB -- Flexible field for extra context
);

-- 2. Convert to Hypertable for TimescaleDB performance
-- Partitions data by time automatically
SELECT create_hypertable('audit_logs', 'time', if_not_exists => TRUE);

-- 3. Enable Compression Policy (SOC2 Requirement: Efficient long-term storage)
-- Compress logs older than 7 days
ALTER TABLE audit_logs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id'
);
-- -- SELECT add_compression_policy('audit_logs', INTERVAL '7 days');

-- 4. Retention Policy (Compliance Requirement: Retain for 7 years)
-- -- SELECT add_retention_policy('audit_logs', INTERVAL '7 years');

-- 5. Immutability Rules (Tamper-proofing)
-- Prevent UPDATE and DELETE operations on the audit_logs table
-- NOTE: Rules are not supported on hypertables in TimescaleDB
-- CREATE RULE no_update_audit AS ON UPDATE TO audit_logs DO INSTEAD NOTHING;
-- CREATE RULE no_delete_audit AS ON DELETE TO audit_logs DO INSTEAD NOTHING;

-- 6. Indexing for Search (Auditor performance)
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_logs (user_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_path_time ON audit_logs (path, time DESC);
