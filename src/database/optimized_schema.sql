-- ============================================================================
-- Black-Scholes Option Pricing Platform - Optimized Database Schema
-- ============================================================================

-- 1. Optimized Data Types and Constraints
ALTER TABLE options_prices 
    ALTER COLUMN implied_volatility TYPE DOUBLE PRECISION,
    ALTER COLUMN delta TYPE DOUBLE PRECISION,
    ALTER COLUMN gamma TYPE DOUBLE PRECISION,
    ALTER COLUMN vega TYPE DOUBLE PRECISION,
    ALTER COLUMN theta TYPE DOUBLE PRECISION,
    ALTER COLUMN rho TYPE DOUBLE PRECISION;

-- 2. TimescaleDB Compression Policy
-- Compress data older than 7 days to save space and improve query performance on historical data
ALTER TABLE options_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('options_prices', INTERVAL '7 days');

-- 3. TimescaleDB Retention Policy
-- Retain raw market data for 2 years
SELECT add_retention_policy('options_prices', INTERVAL '2 years');

-- 4. Continuous Aggregates for Performance
-- Daily OHLC and Analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS options_daily_ohlc
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    strike,
    expiry,
    option_type,
    time_bucket('1 day', time) AS bucket,
    FIRST(last, time) AS open,
    MAX(last) AS high,
    MIN(last) AS low,
    LAST(last, time) AS close,
    SUM(volume) AS total_volume,
    LAST(open_interest, time) AS ending_oi,
    AVG(implied_volatility) AS avg_iv,
    AVG(delta) AS avg_delta
FROM options_prices
GROUP BY symbol, strike, expiry, option_type, bucket;

-- Hourly Greeks for Volatility Surface Construction
CREATE MATERIALIZED VIEW IF NOT EXISTS options_hourly_greeks
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', time) AS bucket,
    strike,
    expiry,
    option_type,
    AVG(delta) AS avg_delta,
    AVG(gamma) AS avg_gamma,
    AVG(vega) AS avg_vega,
    AVG(theta) AS avg_theta,
    AVG(implied_volatility) AS avg_iv
FROM options_prices
WHERE delta IS NOT NULL
GROUP BY symbol, bucket, strike, expiry, option_type;

-- 5. Refresh Policies for Continuous Aggregates
SELECT add_continuous_aggregate_policy('options_daily_ohlc',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('options_hourly_greeks',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes');

-- 6. Strategic Additional Indexes
-- Partial index for open positions (high selectivity)
CREATE INDEX IF NOT EXISTS idx_positions_active_only 
ON positions (portfolio_id, symbol) 
WHERE status = 'open';

-- Index for ML model performance tracking
CREATE INDEX IF NOT EXISTS idx_model_predictions_error_tracking
ON model_predictions (model_id, timestamp DESC)
WHERE actual_price IS NOT NULL;

-- 7. Statistics Tuning
-- Increase statistics target for symbol column for better query planning in large datasets
ALTER TABLE options_prices ALTER COLUMN symbol SET STATISTICS 500;
