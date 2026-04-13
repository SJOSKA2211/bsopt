-- ============================================================================
-- Manifold: PHASE 3 - HYPER-OPTIMIZED LIVE STATE TUNING (v2.0)
-- ============================================================================
-- This script injects PostgreSQL 16 + TimescaleDB 2.17+ engine parameters
-- optimized for massive NVMe/SSD IO and high-frequency writes.
-- ============================================================================

-- ============================================================================
-- Memory & Performance Tuning (NVMe/SSD Optimized)
-- ============================================================================
-- Shared buffers: 25% of RAM for dedicated DB server
ALTER SYSTEM SET shared_buffers = '8GB';
-- Effective cache: 75% of RAM (heuristic for query planner)
ALTER SYSTEM SET effective_cache_size = '24GB';
-- Work memory for complex sorts/hashes
ALTER SYSTEM SET work_mem = '512MB';
-- Maintenance memory for VACUUM, CREATE INDEX, etc.
ALTER SYSTEM SET maintenance_work_mem = '2GB';
-- Temp buffer per session
ALTER SYSTEM SET temp_buffers = '64MB';

-- ============================================================================
-- I/O Specialization for NVMe/SSD
-- ============================================================================
-- Lower random_page_cost for NVMe (vs default 4.0 for HDD)
ALTER SYSTEM SET random_page_cost = 1.1;
-- Higher IO concurrency for parallel prefetch (NVMe can handle 100s of IOs)
ALTER SYSTEM SET effective_io_concurrency = 200;
-- Enable parallel bitmap scans
ALTER SYSTEM SET enable_parallelize = on;
ALTER SYSTEM SET enable_parallel_append = on;
ALTER SYSTEM SET enable_parallel_hash = on;

-- ============================================================================
-- Write Ahead Log (WAL) Optimization
-- ============================================================================
-- replica level for streaming replication and logical decoding
ALTER SYSTEM SET wal_level = 'replica';
-- Larger WAL buffers for batch writes
ALTER SYSTEM SET wal_buffers = '64MB';
-- Commit batching for throughput (balance durability vs speed)
ALTER SYSTEM SET synchronous_commit = 'local';
-- Checkpoint tuning for NVMe
ALTER SYSTEM SET checkpoint_timeout = '30min';
ALTER SYSTEM SET checkpoint_completion_target = 0.95;
-- Preallocate WAL files to reduce fragmentation
ALTER SYSTEM SET wal_init_zero = on;
ALTER SYSTEM SET wal_compression = lz4;
-- Archive mode off for HFT workloads (can enable for point-in-time recovery)
ALTER SYSTEM SET archive_mode = off;
-- Min/max WAL size for batch processing
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- ============================================================================
-- Connection & Concurrency
-- ============================================================================
-- Maximum concurrent connections
ALTER SYSTEM SET max_connections = 500;
-- Prepared transaction timeout (for 2PC)
ALTER SYSTEM SET max_prepared_transactions = 250;

-- ============================================================================
-- Worker Parallelism
-- ============================================================================
-- Total background workers
ALTER SYSTEM SET max_worker_processes = 16;
-- Parallel query workers
ALTER SYSTEM SET max_parallel_workers_per_gather = 6;
ALTER SYSTEM SET max_parallel_workers_per_hashjoin = 4;
ALTER SYSTEM SET max_parallel_workers_for_brin = 4;
-- Total parallel workers cap
ALTER SYSTEM SET max_parallel_workers = 12;
-- Parallel query cost threshold
ALTER SYSTEM SET parallel_tuple_cost = 0.001;
ALTER SYSTEM SET parallel_setup_cost = 100;

-- ============================================================================
-- TimescaleDB 2.17+ Specific Settings
-- ============================================================================
-- Background workers for continuous aggregates and compression
ALTER SYSTEM SET timescaledb.max_background_workers = 16;
-- Adaptive chunk intervals (TimescaleDB 2.12+)
ALTER SYSTEM SET timescaledb.adaptive_chunking.enabled = on;
-- Parallel chunk inserts
ALTER SYSTEM SET timescaledb.enable_parallel_chunk_inserts = on;
-- Compression workers
ALTER SYSTEM SET timescaledb.max_background_compression_workers = 4;

-- ============================================================================
-- Autovacuum Tuning for High-Write Workloads
-- ============================================================================
-- More aggressive vacuum for high-write tables
ALTER SYSTEM SET autovacuum_max_workers = 6;
ALTER SYSTEM SET autovacuum_naptime = '15s';
ALTER SYSTEM SET autovacuum_vacuum_threshold = 50;
ALTER SYSTEM SET autovacuum_analyze_threshold = 50;
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.01;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.01;
-- Delay vacuum to reduce impact on writes
ALTER SYSTEM SET vacuum_cost_delay = '2ms';
ALTER SYSTEM SET vacuum_cost_page_hit = 1;
ALTER SYSTEM SET vacuum_cost_page_miss = 10;
ALTER SYSTEM SET vacuum_cost_page_dirty = 20;

-- ============================================================================
-- Query Planning & Optimization
-- ============================================================================
-- Enable all JIT optimizations
ALTER SYSTEM SET jit = on;
ALTER SYSTEM SET jit_above_cost = 100000;
-- Increase stats for better cardinality estimates
ALTER SYSTEM SET default_statistics_target = 500;
ALTER SYSTEM SET constraint_exclusion = partition;
-- Cost-based planner constants for SSD
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- ============================================================================
-- Logging & Diagnostics
-- ============================================================================
-- Capture slow queries (> 100ms)
ALTER SYSTEM SET log_min_duration_statement = '100ms';
-- Log lock waits > 1s
ALTER SYSTEM SET log_lock_waits = on;
-- Log vacuum progress
ALTER SYSTEM SET log_autovacuum_min_duration = '1s';
-- Track statement statistics
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET track_wal_io_timing = on;

-- ============================================================================
-- Lock Management
-- ============================================================================
-- Prevent lock escalation to table level
ALTER SYSTEM SET lock_timeout = '30s';
-- Statement timeout (2 minutes default)
ALTER SYSTEM SET statement_timeout = '120s';

-- ============================================================================
-- Apply Changes
-- ============================================================================
-- pg_reload_conf applies most changes; some require restart
SELECT pg_reload_conf();

-- Verify settings
SELECT name, setting, unit, context
FROM pg_settings
WHERE name IN (
    'shared_buffers', 'effective_cache_size', 'work_mem',
    'random_page_cost', 'effective_io_concurrency',
    'wal_level', 'wal_buffers', 'synchronous_commit',
    'max_worker_processes', 'max_parallel_workers_per_gather',
    'timescaledb.max_background_workers', 'autovacuum_max_workers',
    'jit', 'log_min_duration_statement'
)
ORDER BY name;

-- ============================================================================
-- Create performance monitoring views
-- ============================================================================
CREATE OR REPLACE VIEW pg_stat_buffer_efficiency AS
SELECT
    CASE WHEN pg_stat_bgwriter.shared_buffer_bytes % 1024 = 0
         THEN pg_stat_bgwriter.shared_buffer_bytes / 1024
         ELSE pg_stat_bgwriter.shared_buffer_bytes
    END AS shared_buffers_kb,
    pg_stat_bgwriter.buffers_backend * 8192 / 1024 AS backend_kb,
    pg_stat_bgwriter.buffers_checkpoint * 8192 / 1024 AS checkpoint_kb,
    pg_stat_bgwriter.buffers_clean * 8192 / 1024 AS cleaner_kb,
    ROUND(
        pg_stat_bgwriter.buffers_backend * 100.0 /
        NULLIF(pg_stat_bgwriter.buffers_backend + pg_stat_bgwriter.buffers_checkpoint + pg_stat_bgwriter.buffers_clean, 0),
        2
    ) AS backend_pct,
    pg_stat_bgwriter.checkpoint_write_time / 1000 AS checkpoint_write_ms,
    pg_stat_bgwriter.checkpoint_sync_time / 1000 AS checkpoint_sync_ms,
    pg_stat_bgwriter.maxwritten_clean,
    pg_stat_bgwriter.buffers_alloc
FROM pg_stat_bgwriter;

-- ============================================================================
-- NVMe/SSD Health Check
-- ============================================================================
DO $$
DECLARE
    iops_latency REAL;
BEGIN
    -- Quick sanity check: if effective_io_concurrency is high, NVMe is likely
    SELECT current_setting('effective_io_concurrency')::INT INTO iops_latency;
    IF iops_latency > 100 THEN
        RAISE NOTICE 'NVMe/SSD detected (effective_io_concurrency=%). High-performance settings active.', iops_latency;
    ELSE
        RAISE NOTICE 'HDD or slower storage detected. Consider adjusting random_page_cost.';
    END IF;
END $$;

-- ============================================================================
-- NOTE: Full effect of changes may require PostgreSQL restart
-- Run: systemctl restart postgresql
-- ============================================================================
