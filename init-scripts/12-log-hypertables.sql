-- 📊 Optimize Audit & Request Logs with TimescaleDB
-- Converts high-volume log tables to hypertables for better performance and management.

CREATE TABLE IF NOT EXISTS request_logs (created_at TIMESTAMPTZ NOT NULL, status_code INT, path TEXT, method TEXT, duration_ms DOUBLE PRECISION);
-- 5. Audit Logs Hypertable
-- Partition by 'created_at' (time)
-- 'id' needs to be removed from PRIMARY KEY constraints before conversion if it exists,
-- but standard TimescaleDB pattern is to rely on time-partitioning.
SELECT create_hypertable('audit_logs', 'created_at', if_not_exists => TRUE);

-- 6. Request Logs Hypertable
SELECT create_hypertable('request_logs', 'created_at', if_not_exists => TRUE);

-- 7. Compression Policies for Logs
-- Compress audit logs after 7 days
ALTER TABLE audit_logs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id',
    timescaledb.compress_orderby = 'created_at DESC'
);
-- -- SELECT add_compression_policy('audit_logs', INTERVAL '7 days');

-- Compress request logs after 3 days
ALTER TABLE request_logs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'status_code',
    timescaledb.compress_orderby = 'created_at DESC'
);
-- -- SELECT add_compression_policy('request_logs', INTERVAL '3 days');

-- 8. Retention Policies for Logs
-- Keep audit logs for 5 years (compliance)
-- -- SELECT add_retention_policy('audit_logs', INTERVAL '5 years');

-- Keep request logs for 30 days (operational debugging)
-- -- SELECT add_retention_policy('request_logs', INTERVAL '30 days');
