-- ============================================================================
-- Black-Scholes Option Pricing Platform - GOD MODE DIAGNOSTICS (PG16)
-- ============================================================================

-- 1. I/O Performance Insight (New in PG16)
-- Helps distinguish between different I/O contexts (bulk vs normal)
CREATE OR REPLACE VIEW pg_stat_io_summary AS
SELECT
    backend_type,
    context,
    object,
    reads,
    read_time,
    writes,
    write_time,
    extends,
    extend_time
FROM pg_stat_io
WHERE reads > 0 OR writes > 0
ORDER BY read_time + write_time DESC;

-- 2. Buffer Usage per Query (Requires pg_stat_statements)
CREATE OR REPLACE VIEW pg_stat_buffer_efficiency AS
SELECT
    query,
    calls,
    shared_blks_hit,
    shared_blks_read,
    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) AS hit_percent
FROM pg_stat_statements
WHERE shared_blks_hit + shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;

-- 3. Connection Tracing (Observe Application Names)
CREATE OR REPLACE VIEW active_connection_audit AS
SELECT
    datname,
    usename,
    application_name,
    client_addr,
    backend_start,
    state,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE state != 'idle';

-- 4. Cumulative Wait Events (Solenya Bottleneck Detector)
CREATE OR REPLACE VIEW system_wait_bottlenecks AS
SELECT
    wait_event_type,
    wait_event,
    count(*) as concurrent_waiters,
    sum(case when state = 'active' then 1 else 0 end) as active_waiters
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
GROUP BY wait_event_type, wait_event
ORDER BY concurrent_waiters DESC;
