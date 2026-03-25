# Database Optimization Architecture (High-Performance)

This document outlines the high-performance PostgreSQL 16 and TimescaleDB 2.17+ architecture implemented for the BS-OPT platform, specifically tuned for a 2GB RAM container environment.

## 1. Engine Configuration (`postgresql.conf`)

The src.shared engine is tuned for maximum throughput within tight memory constraints:
- **Memory Management**:
  - `shared_buffers = 512MB` (25% of RAM)
  - `effective_cache_size = 1536MB` (75% of RAM)
  - `work_mem = 8MB` (Prevents OOM during concurrent complex joins)
- **Parallelism**: Enabled PG16 partition-wise joins and aggregates to leverage multi-src.shared processing for hypertable chunks.
- **Write Optimization**:
  - `checkpoint_timeout = 15min` (Reduces I/O spikes)
  - `autovacuum_vacuum_scale_factor = 0.05` (Aggressive cleaning for high-churn tables)
- **Security**: Enforced `scram-sha-256` password encryption.

## 2. TimescaleDB & Storage Strategy

### Hypertables
All time-series tables (`options_prices`, `market_ticks`, `model_predictions`, `audit_logs`, `request_logs`) are converted to hypertables with optimized chunk intervals.
- **Chunk Skipping**: Enabled on `options_prices` for the `expiry` dimension (TimescaleDB 2.17+).
- **SIMD Compression**:
  - SIMD-vectorized compression enabled on all hypertables and continuous aggregates.
  - `segmentby` columns aligned with query patterns (e.g., `symbol, expiry, strike` for options) to trigger vectorized execution.

### Continuous Aggregates (CAGGs)
A hierarchical refresh strategy ensures real-time analytics with minimal overhead:
1. `minute_stats_cagg`: Base aggregate from raw data.
2. `hourly_stats_chained_cagg`: Chained from minute aggregate.
3. `daily_stats_chained_cagg`: Chained from hourly aggregate.
All CAGGs use compression policies to minimize historical storage footprint.

## 3. Schema & Indexing

- **Data Types**: Transitioned Greeks and Volatility fields to `Double Precision` for faster floating-point math vs `Numeric`.
- **Indexing**:
  - **BRIN Indexes**: Used for large time-ordered tables to save space.
  - **Partial Indexes**: Optimized for common filters like `status = 'open'` or `status_code >= 400`.
  - **Covering Indexes (INCLUDE)**: Enabled Index-Only Scans for the options chain by including bid/ask/Greeks in the index payload.
- **Fill Factor**: Set `FILLFACTOR = 90` for update-heavy tables (`users`, `orders`) to enable HOT updates, and `100` for append-only data.

## 4. Row Level Security (RLS)

Performance-hardened RLS ensures data isolation with zero-leakage:
- Optimized using `EXISTS` logic for relationship checks.
- Uses a stable helper function `get_current_user_id()` which pulls from the session variable `app.current_user_id`.
- Verified isolation between non-superuser roles (`app_user`).

## 5. Application Interaction

- **Connection Pooling**:
  - DB-side: `max_connections = 50`.
  - App-side: `pool_size = 20` with `AsyncAdaptedQueuePool` via SQLAlchemy.
  - This 2:1 ratio ensures the database is never saturated by a single service spike.
- **Pipeliner**: High-speed ingestion uses the **Binary COPY protocol** via `asyncpg` for maximum throughput.

## 6. Observability

- **pg_stat_statements**: Enabled for deep query analysis.
- **Sluggish Query View**: `pg_stat_sluggish_queries` view provided for real-time bottleneck detection.
- **Automated Maintenance**: `make db-optimize` command added for manual `VACUUM` and chunk compression.

---
*Status: Fully Pressurized. High-Performance Active. *
