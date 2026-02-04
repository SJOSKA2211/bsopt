-- 🚀 Optimized Hierarchical Continuous Aggregates
-- Speeds up analytical queries by chaining aggregates.

-- 1. Minute-level stats (Base Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS minute_stats_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 minute', time) AS bucket,
    AVG(last) AS avg_price,
    MAX(last) AS high,
    MIN(last) AS low,
    SUM(volume) AS volume,
    COUNT(*) AS count
FROM options_prices
GROUP BY symbol, bucket
WITH NO DATA;

-- 2. Hourly stats (Chained from Minute Aggregate)
-- Performance optimization: Aggregating from an aggregate is much faster than from raw data.
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats_chained_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 hour', bucket) AS hour,
    AVG(avg_price) AS avg_price,
    MAX(high) AS high,
    MIN(low) AS low,
    SUM(volume) AS volume,
    SUM(count) AS count
FROM minute_stats_cagg
GROUP BY symbol, hour
WITH NO DATA;

-- 3. Daily stats (Chained from Hourly Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats_chained_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', hour) AS day,
    AVG(avg_price) AS avg_price,
    MAX(high) AS high,
    MIN(low) AS low,
    SUM(volume) AS volume,
    SUM(count) AS count
FROM hourly_stats_chained_cagg
GROUP BY symbol, day
WITH NO DATA;

-- 4. Refresh Policies
SELECT add_continuous_aggregate_policy('minute_stats_cagg',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('hourly_stats_chained_cagg',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '10 minutes');

SELECT add_continuous_aggregate_policy('daily_stats_chained_cagg',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- 5. Real-time Greeks Tracking (Approximate via Moving Average)
CREATE MATERIALIZED VIEW IF NOT EXISTS greeks_drift_cagg
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('5 minutes', time) AS bucket,
    AVG(delta) AS avg_delta,
    AVG(gamma) AS avg_gamma,
    AVG(vega) AS avg_vega,
    STDDEV(delta) AS delta_stddev
FROM options_prices -- Assuming delta, gamma, vega columns exist
GROUP BY symbol, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('greeks_drift_cagg',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');
