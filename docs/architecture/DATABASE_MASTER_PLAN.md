# Database Master Plan (High-Performance )

## Overview
This document outlines the "Weaponized" PostgreSQL architecture implemented for the BS-OPT platform, utilizing PostgreSQL 16, TimescaleDB 2.17+, and PGBouncer in a high-throughput transaction mode.

## 1. Engine & Configuration (Production-Ready)
- **Engine**: PostgreSQL 16 with optimized `vacuum_buffer_usage_limit` and `logical_decoding_work_mem`.
- **Commit Strategy**: `synchronous_commit = off` to maximize throughput for HFT-like workloads without sacrificing critical persistence for most time-series data.
- **Memory**: Tuned for 4GB containers with 1GB `shared_buffers` and aggressive `autovacuum` naptime (15s).
- **Security**: Enforced `scram-sha-256` encryption and Implicit Deny in `pg_hba.conf`.

## 2. Connection Management (The Orchestrator)
- **Driver Layer**: 
    - Synchronous: `psycopg` (v3) for optimized prepared statements.
    - Asynchronous: `asyncpg` with high-performance `Binary COPY` support.
- **Pooling**: Adaptive strategy using `NullPool` for transaction-mode PGBouncer and `QueuePool` for direct internal services.
- **Retry Logic**: Implemented in `DatabaseManager` with exponential backoff for transient operational errors.

## 3. Schema & TimescaleDB Optimization
- **Hypertables**: All high-volume tables (`options_prices`, `market_ticks`, `audit_logs`) are hypertables with optimized chunk intervals.
- **Compression**: SIMD-vectorized compression enabled for all time-series data older than 1 day.
- **Continuous Aggregates**: Hierarchical stats (Minute -> Hour -> Day) to offload dashboard query latency.
- **Data Integrity**: Centralized Enum management in `00-extensions.sql` ensuring strict alignment across all services.

## 4. Indexing Strategy
- **BRIN Indexes**: Used for multi-gigabyte time-ordered tables to save 90%+ index space compared to B-tree.
- **INCLUDE Clauses**: Enabled Index-Only Scans for the options chain by including bid/ask/Greeks in the index payload.
- **Partial Indexes**: Optimized for:
    - `orders` (pending/partially_filled only)
    - `positions` (open only)
    - `request_logs` (errors only)

## 5. Observability & Maintenance
- **Diagnostics**: Custom views (`pg_stat_io_summary`, `pg_stat_buffer_efficiency`) for real-time performance monitoring.
- **Benchmarking**: `scripts/benchmark_db.py` provided for quantifying throughput and latency in CI/CD.
- **Maintenance**: Automated `VACUUM` and `ANALYZE` via aggressive autovacuum tuning.

---
*Status: Pressurized. High-Performance Active.*
