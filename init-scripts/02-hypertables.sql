-- ============================================================================
-- Black-Scholes Option Pricing Platform - Optimized Hypertables
-- ============================================================================

-- 1. Options Prices
SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('options_prices', INTERVAL '1 day');
SELECT add_dimension('options_prices', 'expiry', number_partitions => 4, if_not_exists => TRUE);

-- Enable Chunk Skipping (TimescaleDB 2.17+)
DO $$
BEGIN
    PERFORM enable_chunk_skipping('options_prices', 'expiry');
EXCEPTION 
    WHEN undefined_function THEN
        RAISE NOTICE 'enable_chunk_skipping not supported in this TimescaleDB version';
    WHEN duplicate_object THEN
        RAISE NOTICE 'chunk_skipping already enabled for options_prices(expiry)';
    WHEN others THEN
        RAISE NOTICE 'enable_chunk_skipping failed: %', SQLERRM;
END $$;

-- Compression Policy (Compress chunks older than 7 days)
ALTER TABLE options_prices SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('options_prices', INTERVAL '7 days');

-- Retention Policy (Drop chunks older than 2 years)
SELECT add_retention_policy('options_prices', INTERVAL '2 years');

-- 2. Market Ticks
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('market_ticks', INTERVAL '1 hour');
SELECT add_dimension('market_ticks', 'symbol', number_partitions => 8, if_not_exists => TRUE);

-- Compression Policy
ALTER TABLE market_ticks SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('market_ticks', INTERVAL '1 day');

-- Retention Policy
SELECT add_retention_policy('market_ticks', INTERVAL '30 days');

-- 4. Model Predictions
SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);
SELECT set_chunk_time_interval('model_predictions', INTERVAL '1 day');

-- Compression Policy
ALTER TABLE model_predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, model_id', timescaledb.compress_orderby = 'timestamp DESC');
SELECT add_compression_policy('model_predictions', INTERVAL '7 days');

-- 4. Request Logs
SELECT create_hypertable('request_logs', 'created_at', if_not_exists => TRUE);
SELECT set_chunk_time_interval('request_logs', INTERVAL '1 day');

-- Compression Policy
ALTER TABLE request_logs SET (timescaledb.compress, timescaledb.compress_orderby = 'created_at DESC');
SELECT add_compression_policy('request_logs', INTERVAL '1 day');

-- Retention Policy
SELECT add_retention_policy('request_logs', INTERVAL '7 days');

-- 5. Audit Logs (if they exist in 03-audit-schema.sql)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'audit_logs') THEN
        ALTER TABLE audit_logs SET (timescaledb.compress, timescaledb.compress_orderby = 'time DESC');
        PERFORM add_compression_policy('audit_logs', INTERVAL '7 days');
        PERFORM add_retention_policy('audit_logs', INTERVAL '90 days');
    END IF;
END $$;
