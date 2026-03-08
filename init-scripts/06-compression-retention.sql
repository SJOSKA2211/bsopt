-- ============================================================================
-- Black-Scholes Option Pricing Platform - Compression & Retention (GOD MODE)
-- Target: PG 16 + TimescaleDB 2.17+
-- ============================================================================

-- 1. Compression Policies
-- Segment by columns used in frequent WHERE/JOIN clauses to trigger SIMD vectorization

-- options_prices: Optimized for SIMD vectorization & Chunk Skipping
ALTER TABLE options_prices SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'symbol, expiry, strike, option_type', 
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('options_prices', INTERVAL '7 days', if_not_exists => TRUE);

-- market_ticks: Aggressive compression for tick data
ALTER TABLE market_ticks SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'symbol', 
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('market_ticks', INTERVAL '6 hours', if_not_exists => TRUE);

-- audit_logs: Compliance-driven compression
ALTER TABLE audit_logs SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'user_id', 
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('audit_logs', INTERVAL '7 days', if_not_exists => TRUE);

-- request_logs: High-volume request tracking
ALTER TABLE request_logs SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'status_code', 
    timescaledb.compress_orderby = 'created_at DESC'
);
SELECT add_compression_policy('request_logs', INTERVAL '1 day', if_not_exists => TRUE);

-- model_predictions: Data science artifact compression
ALTER TABLE model_predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_id, symbol'
);
SELECT add_compression_policy('model_predictions', INTERVAL '7 days', if_not_exists => TRUE);


-- 2. Retention Policies (Balanced for 4GB environment)
-- Keep core financial data longer, but purge ephemeral logs faster.
SELECT add_retention_policy('options_prices', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('market_ticks', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('audit_logs', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('request_logs', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('model_predictions', INTERVAL '1 year', if_not_exists => TRUE);
