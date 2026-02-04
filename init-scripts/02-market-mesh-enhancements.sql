-- 📊 HYBRID DATA MESH & TFT SUPPORT SCHEMA
-- Optimized for high-frequency price data and SOTA model consumption

-- 1. Create Market Data Mesh Table
-- This table stores normalized data from all sources (Polygon, Scrapers, etc.)
CREATE TABLE IF NOT EXISTS market_data_mesh (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL, -- 'US', 'NSE', 'CRYPTO', etc.
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT,
    source_type TEXT NOT NULL, -- 'api', 'scraper_synthetic', etc.
    metadata JSONB
);

-- 2. Convert to Hypertable for performance
SELECT create_hypertable('market_data_mesh', 'time', if_not_exists => TRUE);

-- 3. Create indices for TFT group_ids lookup
-- TFT requires consistent symbol tagging across time
CREATE INDEX IF NOT EXISTS idx_mesh_symbol_time ON market_data_mesh (symbol, time DESC);

-- 4. Enable compression for long-term storage
ALTER TABLE market_data_mesh SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
-- -- SELECT add_compression_policy('market_data_mesh', INTERVAL '30 days');

-- 5. Continuous Aggregate for OHLCV (Internal use)
-- CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_mesh_daily
-- WITH (timescaledb.continuous) AS
-- SELECT time_bucket('1 day', time) AS bucket,
--        symbol,
--        first(open, time) as open,
--        max(high) as high,
--        min(low) as low,
--        last(close, time) as close,
--        sum(volume) as volume
-- FROM market_data_mesh
-- GROUP BY bucket, symbol;
