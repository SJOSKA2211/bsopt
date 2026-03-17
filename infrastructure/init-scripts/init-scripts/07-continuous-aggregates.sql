-- ============================================================================
-- Black-Scholes Option Pricing Platform - Continuous Aggregates (HIGH-PERFORMANCE)
-- Target: PG 16 + TimescaleDB 2.17+
-- ============================================================================

-- 1. Minute-level stats (Base Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS minute_stats_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 minute', time) AS bucket, AVG(last) AS avg_price, MAX(last) AS high, MIN(last) AS low, SUM(volume) AS volume, COUNT(*) AS count 
FROM options_prices GROUP BY symbol, bucket WITH NO DATA;

SELECT add_continuous_aggregate_policy('minute_stats_cagg', start_offset => INTERVAL '1 hour', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS minute_stats_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('minute_stats_cagg', compress_after => INTERVAL '2 hours', if_not_exists => TRUE);

-- 2. Hourly stats (Chained from Minute Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats_chained_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 hour', bucket) AS hour, AVG(avg_price) AS avg_price, MAX(high) AS high, MIN(low) AS low, SUM(volume) AS volume, SUM(count) AS count 
FROM minute_stats_cagg GROUP BY symbol, hour WITH NO DATA;

SELECT add_continuous_aggregate_policy('hourly_stats_chained_cagg', start_offset => INTERVAL '1 day', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS hourly_stats_chained_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('hourly_stats_chained_cagg', compress_after => INTERVAL '2 days', if_not_exists => TRUE);

-- 3. Daily stats (Chained from Hourly Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats_chained_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 day', hour) AS day, AVG(avg_price) AS avg_price, MAX(high) AS high, MIN(low) AS low, SUM(volume) AS volume, SUM(count) AS count 
FROM hourly_stats_chained_cagg GROUP BY symbol, day WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_stats_chained_cagg', start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS daily_stats_chained_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('daily_stats_chained_cagg', compress_after => INTERVAL '7 days', if_not_exists => TRUE);

-- 4. Real-time Greeks Tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS greeks_drift_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('5 minutes', time) AS bucket, AVG(delta) AS avg_delta, AVG(gamma) AS avg_gamma, AVG(vega) AS avg_vega, STDDEV(delta) AS delta_stddev 
FROM options_prices GROUP BY symbol, bucket WITH NO DATA;

SELECT add_continuous_aggregate_policy('greeks_drift_cagg', start_offset => INTERVAL '6 hours', end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS greeks_drift_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('greeks_drift_cagg', compress_after => INTERVAL '12 hours', if_not_exists => TRUE);

-- 5. Implied Volatility Surface Aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS iv_surface_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('15 minutes', time) AS bucket, AVG(implied_volatility) AS avg_iv, MIN(implied_volatility) AS min_iv, MAX(implied_volatility) AS max_iv, STDDEV(implied_volatility) AS stddev_iv 
FROM options_prices GROUP BY symbol, bucket WITH NO DATA;

SELECT add_continuous_aggregate_policy('iv_surface_cagg', start_offset => INTERVAL '1 day', end_offset => INTERVAL '15 minutes', schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS iv_surface_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('iv_surface_cagg', compress_after => INTERVAL '2 days', if_not_exists => TRUE);

-- 6. Market Ticks OHLCV (1 minute)
CREATE MATERIALIZED VIEW IF NOT EXISTS market_ticks_1m_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 minute', time) AS bucket, 
       first(price, time) as open, MAX(price) as high, MIN(price) as low, last(price, time) as close, SUM(volume) as volume
FROM market_ticks GROUP BY symbol, bucket WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_ticks_1m_cagg', start_offset => INTERVAL '1 hour', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS market_ticks_1m_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('market_ticks_1m_cagg', compress_after => INTERVAL '2 hours', if_not_exists => TRUE);

-- 6b. Market Ticks OHLCV (1 hour, chained)
CREATE MATERIALIZED VIEW IF NOT EXISTS market_ticks_1h_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 hour', bucket) AS hour, 
       first(open, bucket) as open, MAX(high) as high, MIN(low) as low, last(close, bucket) as close, SUM(volume) as volume
FROM market_ticks_1m_cagg GROUP BY symbol, hour WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_ticks_1h_cagg', start_offset => INTERVAL '1 day', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS market_ticks_1h_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('market_ticks_1h_cagg', compress_after => INTERVAL '2 days', if_not_exists => TRUE);

-- 7. Market Data Mesh Daily Aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS market_mesh_daily_cagg WITH (timescaledb.continuous) AS
SELECT symbol, time_bucket('1 day', time) AS bucket, AVG(close) AS avg_close, SUM(volume) AS total_volume, STDDEV(close) as volatility
FROM market_data_mesh GROUP BY symbol, bucket WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_mesh_daily_cagg', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 day', schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW IF EXISTS market_mesh_daily_cagg SET (timescaledb.compress = true);
SELECT add_compression_policy('market_mesh_daily_cagg', compress_after => INTERVAL '14 days', if_not_exists => TRUE);

-- 8. System Metrics Stats (AIOps Monitoring)
DROP MATERIALIZED VIEW IF EXISTS system_metric_stats CASCADE;
CREATE MATERIALIZED VIEW system_metric_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    AVG(latency_ms) as avg_latency,
    STDDEV(latency_ms) as std_latency,
    COUNT(*) as request_count,
    SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as error_count
FROM audit_logs
GROUP BY bucket;

SELECT add_continuous_aggregate_policy('system_metric_stats',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);
