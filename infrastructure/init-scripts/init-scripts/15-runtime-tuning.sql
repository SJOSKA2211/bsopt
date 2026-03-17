-- ============================================================================
-- EQUAFLOW: PHASE 3 - HYPER-OPTIMIZED LIVE STATE TUNING
-- ============================================================================
-- This script injects PostgreSQL engine parameters for massive NVMe/SSD IO.
-- ============================================================================

-- Memory & Performance Tuning
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- I/O Specialization
ALTER SYSTEM SET random_page_cost = 1.0;
ALTER SYSTEM SET effective_io_concurrency = 300;

-- Write Ahead Log (WAL) Optimization
ALTER SYSTEM SET wal_level = 'logical';
ALTER SYSTEM SET wal_buffers = '32MB';
ALTER SYSTEM SET synchronous_commit = 'off';
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Worker Parallelism
ALTER SYSTEM SET max_worker_processes = 12;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- TimescaleDB Specifics
ALTER SYSTEM SET timescaledb.max_background_workers = 16;

-- Apply changes (requires reload/restart, but 'ALTER SYSTEM' persists in postgresql.auto.conf)
SELECT pg_reload_conf();
