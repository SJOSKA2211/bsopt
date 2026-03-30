-- ============================================================================
-- Continuous Aggregates for OHLCV Data
-- ============================================================================
-- Creates materialized views for real-time aggregation:
-- 1 minute -> 5 minute -> 15 minute -> 1 hour -> 1 day
-- ============================================================================

-- Ensure TimescaleDB is available
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RAISE NOTICE 'TimescaleDB extension not available. Continuous aggregates will not be created.';
    END IF;
END $$;

-- ============================================================================
-- 1 Minute OHLCV Aggregate
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS volume,
       avg(price) AS vwap,
       count(*) AS tick_count
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- Add refresh policy for 1m aggregates
DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_1m',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 minute',
        schedule_interval => INTERVAL '1 minute');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for ohlcv_1m skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- 5 Minute OHLCV Aggregate
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT time_bucket('5 minutes', time) AS bucket,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS volume,
       avg(price) AS vwap,
       stddev(price) AS price_stddev,
       count(*) AS tick_count
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_5m',
        start_offset => INTERVAL '12 hours',
        end_offset => INTERVAL '5 minutes',
        schedule_interval => INTERVAL '5 minutes');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for ohlcv_5m skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- 15 Minute OHLCV Aggregate
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15m
WITH (timescaledb.continuous) AS
SELECT time_bucket('15 minutes', time) AS bucket,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS volume,
       avg(price) AS vwap,
       stddev(price) AS price_stddev,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
       count(*) AS tick_count
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_15m',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '15 minutes',
        schedule_interval => INTERVAL '15 minutes');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for ohlcv_15m skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- 1 Hour OHLCV Aggregate
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS bucket,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS volume,
       avg(price) AS vwap,
       stddev(price) AS price_stddev,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY price) AS q1_price,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY price) AS q3_price,
       count(*) AS tick_count,
       count(DISTINCT market) AS market_count
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_1h',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for ohlcv_1h skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- 1 Day OHLCV Aggregate
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1d
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', time) AS bucket,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS total_volume,
       avg(price) AS vwap,
       stddev(price) AS price_stddev,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY price) AS q1_price,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY price) AS q3_price,
       count(*) AS tick_count,
       count(DISTINCT market) AS market_count,
       array_agg(DISTINCT market) AS markets
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('ohlcv_1d',
        start_offset => INTERVAL '90 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for ohlcv_1d skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- Volatility Metrics by Symbol
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS symbol_volatility
WITH (timescaledb.continuous) AS
SELECT time_bucket('5 minutes', time) AS bucket,
       symbol,
       stddev(price) / avg(price) AS coefficient_of_variation,
       max(price) - min(price) AS intraday_range,
       (max(price) - min(price)) / min(price) * 100 AS intraday_range_pct,
       stddev(price) AS price_volatility,
       avg(price) AS mean_price,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY price) AS var_99,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY price) AS var_95
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy('symbol_volatility',
        start_offset => INTERVAL '12 hours',
        end_offset => INTERVAL '5 minutes',
        schedule_interval => INTERVAL '5 minutes');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate policy for symbol_volatility skipped: %', SQLERRM;
END $$;

-- ============================================================================
-- Refresh all continuous aggregates
-- ============================================================================
SELECT refresh_continuous_aggregate('ohlcv_1m', NULL, NULL);
SELECT refresh_continuous_aggregate('ohlcv_5m', NULL, NULL);
SELECT refresh_continuous_aggregate('ohlcv_15m', NULL, NULL);
SELECT refresh_continuous_aggregate('ohlcv_1h', NULL, NULL);
SELECT refresh_continuous_aggregate('ohlcv_1d', NULL, NULL);
SELECT refresh_continuous_aggregate('symbol_volatility', NULL, NULL);

-- ============================================================================
-- Indexes for Continuous Aggregates
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_time ON ohlcv_1m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_symbol_time ON ohlcv_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_symbol_time ON ohlcv_15m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1h_symbol_time ON ohlcv_1h (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1d_symbol_time ON ohlcv_1d (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_vol_symbol_time ON symbol_volatility (symbol, bucket DESC);

-- ============================================================================
-- Grant permissions
-- ============================================================================
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT, UPDATE ON ALL TABLES IN SCHEMA public TO readwrite_user;
