-- ============================================================================
-- Black-Scholes Option Pricing Platform - Hypertables
-- ============================================================================

SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('options_prices', INTERVAL '1 day');
SELECT add_dimension('options_prices', 'expiry', number_partitions => 4, if_not_exists => TRUE);

SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('market_ticks', INTERVAL '1 hour');
SELECT add_dimension('market_ticks', 'symbol', number_partitions => 8, if_not_exists => TRUE);

SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);
SELECT set_chunk_time_interval('model_predictions', INTERVAL '1 day');

SELECT create_hypertable('request_logs', 'created_at', if_not_exists => TRUE);
