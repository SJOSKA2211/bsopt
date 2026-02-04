-- 🚀 TimescaleDB Continuous Aggregates for High-Performance Analytics
-- Replaces traditional Materialized Views with automatically refreshing aggregates.

-- 1. Daily Market OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', time) AS day,
    FIRST(last, time) AS open,
    MAX(last) AS high,
    MIN(last) AS low,
    LAST(last, time) AS close,
    SUM(volume) AS volume
FROM options_prices
GROUP BY symbol, day
WITH NO DATA;

-- Enable real-time aggregation and set compression segments
ALTER MATERIALIZED VIEW daily_ohlcv_cagg SET (
    timescaledb.materialized_only = false,
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'day DESC'
);

-- 2. Hourly Market Stats
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 hour', time) AS hour,
    AVG(last) AS avg_price,
    SUM(volume) AS total_volume,
    COUNT(*) AS tick_count
FROM options_prices
GROUP BY symbol, hour
WITH NO DATA;

-- Enable real-time aggregation and set compression segments
ALTER MATERIALIZED VIEW hourly_stats_cagg SET (
    timescaledb.materialized_only = false,
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'hour DESC'
);

-- 3. Implied Volatility Surface Aggregates (New Optimization)
-- Aggregates IV by 15-minute buckets for surface construction and historical analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS iv_surface_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('15 minutes', time) AS bucket,
    AVG(implied_volatility) AS avg_iv,
    MIN(implied_volatility) AS min_iv,
    MAX(implied_volatility) AS max_iv,
    STDDEV(implied_volatility) AS stddev_iv
FROM options_prices
GROUP BY symbol, bucket
WITH NO DATA;

-- Enable real-time aggregation and set compression segments for IV
ALTER MATERIALIZED VIEW iv_surface_cagg SET (
    timescaledb.materialized_only = false,
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'bucket DESC'
);

-- 4. Refresh Policies
-- Refresh daily aggregate every hour for the last 3 days
SELECT add_continuous_aggregate_policy('daily_ohlcv_cagg',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Refresh hourly aggregate every 15 minutes
SELECT add_continuous_aggregate_policy('hourly_stats_cagg',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes');

-- Refresh IV aggregate every 15 minutes
SELECT add_continuous_aggregate_policy('iv_surface_cagg',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes');