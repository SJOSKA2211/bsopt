-- ============================================================================
-- Black-Scholes Option Pricing Platform - Hypertables
-- ============================================================================

-- 1. Options Prices
SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('options_prices', INTERVAL '1 day');
SELECT add_dimension('options_prices', 'expiry', number_partitions => 4, if_not_exists => TRUE);

-- Enable Chunk Skipping (TimescaleDB 2.17+)
DO $$
BEGIN
    PERFORM enable_chunk_skipping('options_prices', 'expiry');
EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'enable_chunk_skipping not supported in this TimescaleDB version';
END $$;

-- 2. Market Ticks
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('market_ticks', INTERVAL '1 hour');
SELECT add_dimension('market_ticks', 'symbol', number_partitions => 8, if_not_exists => TRUE);

SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);
SELECT set_chunk_time_interval('model_predictions', INTERVAL '1 day');

-- 3. Request Logs
SELECT create_hypertable('request_logs', 'created_at', if_not_exists => TRUE);
SELECT set_chunk_time_interval('request_logs', INTERVAL '1 day');
