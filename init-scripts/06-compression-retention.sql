-- ============================================================================
-- Black-Scholes Option Pricing Platform - Compression & Retention
-- ============================================================================

-- Compression Policies
ALTER TABLE options_prices SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('options_prices', INTERVAL '1 day', if_not_exists => TRUE);

ALTER TABLE market_ticks SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('market_ticks', INTERVAL '6 hours', if_not_exists => TRUE);

ALTER TABLE market_data_mesh SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('market_data_mesh', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE audit_logs SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('audit_logs', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE request_logs SET (timescaledb.compress, timescaledb.compress_segmentby = 'status_code', timescaledb.compress_orderby = 'created_at DESC');
SELECT add_compression_policy('request_logs', INTERVAL '3 days', if_not_exists => TRUE);


-- Retention Policies
SELECT add_retention_policy('options_prices', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('market_ticks', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('market_data_mesh', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('audit_logs', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('request_logs', INTERVAL '30 days', if_not_exists => TRUE);
