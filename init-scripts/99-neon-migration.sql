-- ============================================================================
-- Neon Serverless Postgres Migration (standardizing from TimescaleDB)
-- ============================================================================

-- 1. Recreate options_prices with standard partitioning
-- Note: We assume the original table might exist but we need to convert it.
-- In a real migration, we would rename the old table, create the new one, and move data.

CREATE TABLE IF NOT EXISTS options_prices_neon (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strike NUMERIC(12, 2) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    bid NUMERIC(12, 4),
    ask NUMERIC(12, 4),
    last NUMERIC(12, 4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC(8, 6),
    delta NUMERIC(8, 6),
    gamma NUMERIC(8, 6),
    vega NUMERIC(8, 6),
    theta NUMERIC(8, 6),
    rho NUMERIC(8, 6),
    underlying_price NUMERIC(12, 4),
    metadata JSONB
) PARTITION BY RANGE (time);

-- Create initial partitions (e.g., for early 2026)
CREATE TABLE IF NOT EXISTS options_prices_y2026m02 PARTITION OF options_prices_neon
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- 2. Standard Materialized Views (replacing Continuous Aggregates)

CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_standard AS
SELECT 
    symbol,
    date_trunc('day', time) AS day,
    (array_agg(last ORDER BY time ASC))[1] AS open,
    MAX(last) AS high,
    MIN(last) AS low,
    (array_agg(last ORDER BY time DESC))[1] AS close,
    SUM(volume) AS volume
FROM options_prices_neon
GROUP BY symbol, day;

CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_ohlcv_standard_symbol_day ON daily_ohlcv_standard(symbol, day);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats_standard AS
SELECT 
    symbol,
    date_trunc('hour', time) AS hour,
    AVG(last) AS avg_price,
    SUM(volume) AS total_volume,
    COUNT(*) AS tick_count
FROM options_prices_neon
GROUP BY symbol, hour;

CREATE UNIQUE INDEX IF NOT EXISTS idx_hourly_stats_standard_symbol_hour ON hourly_stats_standard(symbol, hour);

-- 3. Utility function for refreshing views (since Neon doesn't have background workers like Timescale)

CREATE OR REPLACE FUNCTION refresh_market_views() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_ohlcv_standard;
    REFRESH MATERIALIZED VIEW CONCURRENTLY hourly_stats_standard;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_market_views() IS 'Standard refresh for Neon serverless views.';
