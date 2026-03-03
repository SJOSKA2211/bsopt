-- ============================================================================
-- Black-Scholes Option Pricing Platform - Audit Logs
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    time TIMESTAMPTZ NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    client_ip TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    metadata JSONB
);

SELECT create_hypertable('audit_logs', 'time', if_not_exists => TRUE);
