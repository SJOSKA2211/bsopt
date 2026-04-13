-- ============================================================================
-- Black-Scholes Option Pricing Platform - Optimized Hypertables
-- ============================================================================

-- 1. Options Prices
SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('options_prices', INTERVAL '1 day');
SELECT add_dimension('options_prices', 'symbol', number_partitions => 8, if_not_exists => TRUE);

-- Enable Chunk Skipping (TimescaleDB 2.17+)
DO $$
BEGIN
    PERFORM enable_chunk_skipping('options_prices', 'expiry');
EXCEPTION WHEN others THEN
    RAISE NOTICE 'enable_chunk_skipping skipped or failed: %', SQLERRM;
END $$;

-- Compression Policy (Compress chunks older than 7 days)
DO $$
BEGIN
    ALTER TABLE options_prices SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
EXCEPTION WHEN others THEN
    RAISE NOTICE 'compression settings already applied or failed: %', SQLERRM;
END $$;

DO $$
BEGIN
    PERFORM add_compression_policy('options_prices', INTERVAL '7 days');
EXCEPTION WHEN others THEN
    RAISE NOTICE 'compression policy already exists or failed: %', SQLERRM;
END $$;

-- Retention Policy (Drop chunks older than 2 years)
DO $$
BEGIN
    PERFORM add_retention_policy('options_prices', INTERVAL '2 years');
EXCEPTION WHEN others THEN
    RAISE NOTICE 'retention policy already exists or failed: %', SQLERRM;
END $$;

-- 2. Market Ticks
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('market_ticks', INTERVAL '1 hour');
SELECT add_dimension('market_ticks', 'symbol', number_partitions => 8, if_not_exists => TRUE);

-- Compression Policy
DO $$
BEGIN
    ALTER TABLE market_ticks SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
EXCEPTION WHEN others THEN NULL;
END $$;

DO $$
BEGIN
    PERFORM add_compression_policy('market_ticks', INTERVAL '1 day');
EXCEPTION WHEN others THEN NULL;
END $$;

-- Retention Policy
DO $$
BEGIN
    PERFORM add_retention_policy('market_ticks', INTERVAL '30 days');
EXCEPTION WHEN others THEN NULL;
END $$;

-- Model Predictions now uses Native Partitioning defined in 01-core-schema.sql

-- 4. Request Logs
SELECT create_hypertable('request_logs', 'created_at', if_not_exists => TRUE);
SELECT set_chunk_time_interval('request_logs', INTERVAL '1 day');

-- Compression Policy
DO $$
BEGIN
    ALTER TABLE request_logs SET (timescaledb.compress, timescaledb.compress_orderby = 'created_at DESC');
EXCEPTION WHEN others THEN NULL;
END $$;

DO $$
BEGIN
    PERFORM add_compression_policy('request_logs', INTERVAL '1 day');
EXCEPTION WHEN others THEN NULL;
END $$;

-- Retention Policy
DO $$
BEGIN
    PERFORM add_retention_policy('request_logs', INTERVAL '7 days');
EXCEPTION WHEN others THEN NULL;
END $$;

-- 5. Audit Logs (if they exist in 03-audit-schema.sql)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'audit_logs') THEN
        BEGIN
            ALTER TABLE audit_logs SET (timescaledb.compress, timescaledb.compress_orderby = 'time DESC');
        EXCEPTION WHEN others THEN NULL;
        END;
        
        BEGIN
            PERFORM add_compression_policy('audit_logs', INTERVAL '7 days');
        EXCEPTION WHEN others THEN NULL;
        END;
        
        BEGIN
            PERFORM add_retention_policy('audit_logs', INTERVAL '90 days');
        EXCEPTION WHEN others THEN NULL;
        END;
    END IF;
END $$;
