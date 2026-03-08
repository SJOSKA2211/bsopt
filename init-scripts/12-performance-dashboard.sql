-- ============================================================================
-- Black-Scholes Option Pricing Platform - Performance Dashboard Views
-- ============================================================================

DROP VIEW IF EXISTS db_health_overview CASCADE;
DROP VIEW IF EXISTS query_variance_report CASCADE;
DROP VIEW IF EXISTS cache_hit_ratio CASCADE;
DROP VIEW IF EXISTS index_efficiency_audit CASCADE;
DROP VIEW IF EXISTS transaction_throughput CASCADE;
DROP VIEW IF EXISTS lock_contention_summary CASCADE;
DROP VIEW IF EXISTS pg_stat_sluggish_queries CASCADE;

-- 1. Combined Health Overview
CREATE OR REPLACE VIEW db_health_overview AS
SELECT
    current_database() AS db_name,
    version() AS pg_version,
    (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') AS timescaledb_version,
    (SELECT count(*) FROM timescaledb_information.hypertables) AS hypertables_count,
    (SELECT count(*) FROM timescaledb_information.continuous_aggregates) AS caggs_count,
    (SELECT count(*) FROM pg_stat_activity) AS active_connections,
    (SELECT pg_size_pretty(pg_database_size(current_database()))) AS db_size;

-- 2. Query Performance Heatmap (Last 1000 Queries)
-- Focuses on execution time variance
CREATE OR REPLACE VIEW query_variance_report AS
SELECT 
    query, 
    calls, 
    round(total_exec_time::numeric, 2) as total_ms,
    round(mean_exec_time::numeric, 2) as mean_ms,
    round(stddev_exec_time::numeric, 2) as stddev_ms,
    round((stddev_exec_time / NULLIF(mean_exec_time, 0))::numeric, 4) as coefficient_of_variation
FROM pg_stat_statements
WHERE calls > 5
ORDER BY stddev_ms DESC
LIMIT 20;

-- 3. Cache Hit Ratio (The "Memory Pressure" Gauge)
CREATE OR REPLACE VIEW cache_hit_ratio AS
SELECT 
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1) AS hit_ratio
FROM pg_statio_user_tables;

-- 4. Index Efficiency Audit (Unused vs. Scanned)
CREATE OR REPLACE VIEW index_efficiency_audit AS
SELECT 
    relname AS table_name, 
    indexrelname AS index_name, 
    idx_scan AS scan_count, 
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes 
WHERE idx_scan < 10 AND pg_relation_size(indexrelid) > 1024 * 1024
ORDER BY pg_relation_size(indexrelid) DESC;

-- 5. Transaction Throughput (Since Last Reset)
CREATE OR REPLACE VIEW transaction_throughput AS
SELECT 
    datname, 
    xact_commit, 
    xact_rollback, 
    round(xact_commit::numeric / (xact_commit + xact_rollback + 1), 4) as success_rate,
    stats_reset
FROM pg_stat_database 
WHERE datname = current_database();

-- 6. Lock Contention Detector
CREATE OR REPLACE VIEW lock_contention_summary AS
SELECT
    blocked_locks.pid     AS blocked_pid,
    blocking_locks.pid    AS blocking_pid,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM  pg_catalog.pg_locks         blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity  ON blocked_locks.pid = blocked_activity.pid
JOIN pg_catalog.pg_locks         blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_locks.pid = blocking_activity.pid
WHERE NOT blocked_locks.GRANTED;

-- 7. Sluggish Query Detector
-- Focuses on high average execution time
CREATE OR REPLACE VIEW pg_stat_sluggish_queries AS
SELECT 
    query, 
    calls, 
    round(mean_exec_time::numeric, 2) as mean_ms,
    round(total_exec_time::numeric, 2) as total_ms,
    rows
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- 8. TimescaleDB Background Job Audit
CREATE OR REPLACE VIEW job_performance_audit AS
SELECT
    j.job_id,
    proc_name,
    schedule_interval,
    last_run_started_at,
    last_run_duration,
    last_run_status,
    total_runs,
    total_failures
FROM timescaledb_information.jobs j
JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id;
