-- 🚀 SINGULARITY: Absolute TimescaleDB Optimization
-- Enforces tiered storage, continuous aggregation, and zero-latency refresh policies.

-- 1. Enable Continuous Aggregates
-- Ensure the daily OHLCV is active
ALTER MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_cagg SET (timescaledb.materialized_only = false);

-- 2. Configure Automatic Refresh Policies
-- Refresh daily aggregate every 1 hour, looking back 3 days for late data
SELECT add_continuous_aggregate_policy('daily_ohlcv_cagg',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- 3. Enforce Compression Policies (SOTA)
-- Compress options prices older than 1 day
SELECT add_compression_policy('options_prices', INTERVAL '1 day', if_not_exists => TRUE);

-- Compress market ticks older than 6 hours (high volume)
SELECT add_compression_policy('market_ticks', INTERVAL '6 hours', if_not_exists => TRUE);

-- 4. Tiered Storage: Chunk Mapping
-- Ensure chunks are sized for the L3 cache of the Neon storage nodes (approx 100MB per chunk)
SELECT set_chunk_time_interval('options_prices', INTERVAL '1 day');
SELECT set_chunk_time_interval('market_ticks', INTERVAL '1 hour');

-- 5. Enable Real-time Aggregates for IV
ALTER MATERIALIZED VIEW IF NOT EXISTS iv_surface_cagg SET (timescaledb.materialized_only = false);
SELECT add_continuous_aggregate_policy('iv_surface_cagg',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE);
