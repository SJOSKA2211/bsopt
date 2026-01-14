-- ============================================================================
-- Black-Scholes Option Pricing Platform - Database Optimization Script
-- ============================================================================
--
-- This script implements all database optimizations recommended in the
-- performance optimization report.
--
-- Optimizations Include:
-- 1. Strategic indexes for query performance
-- 2. TimescaleDB continuous aggregates
-- 3. Compression policies
-- 4. Retention policies
-- 5. Partition optimization
--
-- Expected Performance Impact:
-- - Query latency: 50-100ms → 5-10ms (5-10x improvement)
-- - Storage usage: -80% (compression + retention)
-- - Analytics queries: 100-500ms → 5-20ms (10-50x improvement)
--
-- ============================================================================

-- Set search path
SET search_path TO public;

-- ============================================================================
-- PART 1: STRATEGIC INDEXES
-- ============================================================================

-- Drop existing problematic indexes if they exist
-- (Run this section only if migrating from suboptimal indexes)
-- DROP INDEX IF EXISTS idx_options_prices_symbol_time;
-- DROP INDEX IF EXISTS idx_options_prices_expiry_time;

COMMENT ON TABLE options_prices IS 'High-frequency options market data (TimescaleDB hypertable)';

-- Index 1: Symbol + Time Descending (for latest price lookups)
-- Covers queries: SELECT * FROM options_prices WHERE symbol = 'AAPL' ORDER BY time DESC LIMIT 10
CREATE INDEX IF NOT EXISTS idx_options_prices_symbol_time_desc
ON options_prices (symbol, time DESC);

COMMENT ON INDEX idx_options_prices_symbol_time_desc IS
'Optimized for recent price lookups by symbol';

-- Index 2: Option Chain Lookup (symbol, expiry, strike, type, time)
-- Covers queries: SELECT * FROM options_prices WHERE symbol = 'AAPL' AND expiry = '2024-12-20'
CREATE INDEX IF NOT EXISTS idx_options_prices_chain_lookup
ON options_prices (symbol, expiry, strike, option_type, time DESC);

COMMENT ON INDEX idx_options_prices_chain_lookup IS
'Optimized for option chain construction';

-- Index 3: Greeks Availability (for filtering options with calculated Greeks)
-- Covers queries: SELECT * FROM options_prices WHERE delta IS NOT NULL
CREATE INDEX IF NOT EXISTS idx_options_prices_greeks_available
ON options_prices (symbol, time DESC)
WHERE delta IS NOT NULL AND gamma IS NOT NULL;

COMMENT ON INDEX idx_options_prices_greeks_available IS
'Optimized for filtering options with available Greeks';

-- Index 4: Implied Volatility Analysis
-- Covers queries: SELECT * FROM options_prices WHERE implied_volatility BETWEEN 0.2 AND 0.4
CREATE INDEX IF NOT EXISTS idx_options_prices_iv_analysis
ON options_prices (symbol, expiry, implied_volatility, time DESC)
WHERE implied_volatility IS NOT NULL;

COMMENT ON INDEX idx_options_prices_iv_analysis IS
'Optimized for implied volatility surface construction';

-- ============================================================================
-- Index Optimization for Portfolios & Positions
-- ============================================================================

-- Index 5: Portfolio Positions (active positions)
-- Covers queries: SELECT * FROM positions WHERE portfolio_id = ? AND status = 'open'
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status_open
ON positions (portfolio_id, status, entry_date DESC)
WHERE status = 'open';

COMMENT ON INDEX idx_positions_portfolio_status_open IS
'Optimized for active position lookups';

-- Index 6: Expiring Positions (for monitoring and alerts)
-- Covers queries: SELECT * FROM positions WHERE expiry <= NOW() + INTERVAL '7 days' AND status = 'open'
CREATE INDEX IF NOT EXISTS idx_positions_expiring_soon
ON positions (expiry, status)
WHERE status = 'open' AND expiry IS NOT NULL;

COMMENT ON INDEX idx_positions_expiring_soon IS
'Optimized for expiration monitoring';

-- ============================================================================
-- Index Optimization for Orders
-- ============================================================================

-- Index 7: Recent User Orders
-- Covers queries: SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT 100
CREATE INDEX IF NOT EXISTS idx_orders_user_recent
ON orders (user_id, created_at DESC, status);

COMMENT ON INDEX idx_orders_user_recent IS
'Optimized for recent order history';

-- Index 8: Pending Orders (for order execution monitoring)
-- Covers queries: SELECT * FROM orders WHERE status IN ('pending', 'partially_filled')
CREATE INDEX IF NOT EXISTS idx_orders_pending
ON orders (status, created_at DESC)
WHERE status IN ('pending', 'partially_filled');

COMMENT ON INDEX idx_orders_pending IS
'Optimized for active order monitoring';

-- Index 9: Broker Order ID Lookup
-- Covers queries: SELECT * FROM orders WHERE broker = 'interactive_brokers' AND broker_order_id = ?
CREATE INDEX IF NOT EXISTS idx_orders_broker_lookup
ON orders (broker, broker_order_id)
WHERE broker_order_id IS NOT NULL;

COMMENT ON INDEX idx_orders_broker_lookup IS
'Optimized for broker order reconciliation';

-- ============================================================================
-- Index Optimization for ML Models & Predictions
-- ============================================================================

-- Index 10: Production Model Lookup
-- Covers queries: SELECT * FROM ml_models WHERE name = 'iv_predictor' AND is_production = true
CREATE INDEX IF NOT EXISTS idx_ml_models_production_lookup
ON ml_models (name, is_production, version DESC)
WHERE is_production = true;

COMMENT ON INDEX idx_ml_models_production_lookup IS
'Optimized for production model lookups';

-- Index 11: Pending Predictions (for backtesting)
-- Covers queries: SELECT * FROM model_predictions WHERE actual_price IS NULL
CREATE INDEX IF NOT EXISTS idx_model_predictions_pending
ON model_predictions (model_id, timestamp DESC)
WHERE actual_price IS NULL;

COMMENT ON INDEX idx_model_predictions_pending IS
'Optimized for pending prediction lookups';

-- ============================================================================
-- Index Maintenance
-- ============================================================================

-- Analyze tables to update statistics
ANALYZE options_prices;
ANALYZE positions;
ANALYZE orders;
ANALYZE ml_models;
ANALYZE model_predictions;

-- Check index sizes
SELECT
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;

-- ============================================================================
-- PART 2: TIMESCALEDB CONTINUOUS AGGREGATES
-- ============================================================================

-- Ensure TimescaleDB extension is enabled
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Adjust model_predictions primary key to include timestamp (required for hypertable)
ALTER TABLE model_predictions DROP CONSTRAINT IF EXISTS model_predictions_pkey;
ALTER TABLE model_predictions ADD PRIMARY KEY (id, timestamp);

-- Convert model_predictions to hypertable if not already
SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);

-- Continuous Aggregate 1: Daily OHLC (Open, High, Low, Close)
-- Purpose: Pre-computed daily summaries for historical analysis
DROP MATERIALIZED VIEW IF EXISTS options_daily_ohlc CASCADE;

CREATE MATERIALIZED VIEW options_daily_ohlc
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    strike,
    expiry,
    option_type,
    time_bucket('1 day', time) AS bucket,
    FIRST(last, time) AS open,
    MAX(last) AS high,
    MIN(last) AS low,
    LAST(last, time) AS close,
    SUM(volume) AS total_volume,
    LAST(open_interest, time) AS ending_oi,
    AVG(implied_volatility) AS avg_iv,
    COUNT(*) AS sample_count
FROM options_prices
WHERE last IS NOT NULL
GROUP BY symbol, strike, expiry, option_type, bucket;

-- Create index on continuous aggregate
CREATE INDEX IF NOT EXISTS idx_options_daily_ohlc_symbol_bucket
ON options_daily_ohlc (symbol, bucket DESC);

-- Add refresh policy: refresh every hour for last 3 days
SELECT add_continuous_aggregate_policy('options_daily_ohlc',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

COMMENT ON TABLE options_daily_ohlc IS
'Daily OHLC summaries for options prices (auto-refreshed hourly)';

-- Continuous Aggregate 2: Hourly Greeks Statistics
-- Purpose: Pre-computed Greeks for volatility surface construction
DROP MATERIALIZED VIEW IF EXISTS options_hourly_greeks CASCADE;

CREATE MATERIALIZED VIEW options_hourly_greeks
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', time) AS bucket,
    strike,
    expiry,
    option_type,
    AVG(delta) AS avg_delta,
    STDDEV(delta) AS stddev_delta,
    AVG(gamma) AS avg_gamma,
    AVG(vega) AS avg_vega,
    AVG(theta) AS avg_theta,
    AVG(implied_volatility) AS avg_iv,
    STDDEV(implied_volatility) AS stddev_iv,
    AVG(last) AS avg_price,
    COUNT(*) AS sample_count
FROM options_prices
WHERE delta IS NOT NULL AND implied_volatility IS NOT NULL
GROUP BY symbol, bucket, strike, expiry, option_type;

-- Create index on continuous aggregate
CREATE INDEX IF NOT EXISTS idx_options_hourly_greeks_symbol_bucket
ON options_hourly_greeks (symbol, bucket DESC);

-- Add refresh policy: refresh every 30 minutes for last 2 days
SELECT add_continuous_aggregate_policy('options_hourly_greeks',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE);

COMMENT ON TABLE options_hourly_greeks IS
'Hourly Greeks statistics for volatility surface (auto-refreshed every 30 min)';

-- Continuous Aggregate 3: Model Performance Metrics (Daily)
-- Purpose: Track ML model prediction accuracy over time
DROP MATERIALIZED VIEW IF EXISTS model_daily_performance CASCADE;

CREATE MATERIALIZED VIEW model_daily_performance
WITH (timescaledb.continuous) AS
SELECT
    model_id,
    time_bucket('1 day', timestamp) AS bucket,
    COUNT(*) AS prediction_count,
    COUNT(*) FILTER (WHERE actual_price IS NOT NULL) AS evaluated_count,
    AVG(ABS(prediction_error)) AS mae,
    SQRT(AVG(prediction_error * prediction_error)) AS rmse,
    AVG(prediction_error) AS bias,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(prediction_error)) AS median_error,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(prediction_error)) AS p95_error,
    MAX(ABS(prediction_error)) AS max_error
FROM model_predictions
WHERE actual_price IS NOT NULL AND prediction_error IS NOT NULL
GROUP BY model_id, bucket;

-- Create index on continuous aggregate
CREATE INDEX IF NOT EXISTS idx_model_daily_performance_model_bucket
ON model_daily_performance (model_id, bucket DESC);

-- Add refresh policy: refresh every 6 hours
SELECT add_continuous_aggregate_policy('model_daily_performance',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE);

COMMENT ON TABLE model_daily_performance IS
'Daily ML model performance metrics (auto-refreshed every 6 hours)';

-- ============================================================================
-- PART 3: COMPRESSION POLICIES
-- ============================================================================

-- Enable compression on options_prices hypertable (compress data older than 7 days)
-- Expected compression ratio: 10-20x
-- Storage savings: 500GB/year → 25-50GB/year
SELECT add_compression_policy('options_prices', INTERVAL '7 days', if_not_exists => TRUE);

-- Set compression segments for optimal query performance
ALTER TABLE options_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, strike, expiry, option_type',
    timescaledb.compress_orderby = 'time DESC'
);

COMMENT ON TABLE options_prices IS
'TimescaleDB hypertable with compression enabled (7-day threshold)';

-- ============================================================================
-- PART 4: RETENTION POLICIES
-- ============================================================================

-- Drop raw data older than 2 years (keep continuous aggregates longer)
SELECT add_retention_policy('options_prices', INTERVAL '2 years', if_not_exists => TRUE);

-- Keep continuous aggregates longer
-- Daily OHLC: 5 years
-- SELECT add_retention_policy('options_daily_ohlc', INTERVAL '5 years');

-- Hourly Greeks: 3 years
-- SELECT add_retention_policy('options_hourly_greeks', INTERVAL '3 years');

-- Model predictions: 1 year for raw data
SELECT add_retention_policy('model_predictions', INTERVAL '1 year', if_not_exists => TRUE);

COMMENT ON TABLE options_prices IS
'TimescaleDB hypertable with 2-year retention policy';

-- ============================================================================
-- PART 5: QUERY PERFORMANCE VALIDATION
-- ============================================================================

-- Test Query 1: Latest prices by symbol (should use idx_options_prices_symbol_time_desc)
EXPLAIN ANALYZE
SELECT * FROM options_prices
WHERE symbol = 'AAPL'
ORDER BY time DESC
LIMIT 10;

-- Test Query 2: Option chain lookup (should use idx_options_prices_chain_lookup)
EXPLAIN ANALYZE
SELECT * FROM options_prices
WHERE symbol = 'AAPL'
  AND expiry = CURRENT_DATE + INTERVAL '30 days'
ORDER BY strike, time DESC;

-- Test Query 3: Active positions (should use idx_positions_portfolio_status_open)
EXPLAIN ANALYZE
SELECT * FROM positions
WHERE portfolio_id = (SELECT id FROM portfolios LIMIT 1)
  AND status = 'open'
ORDER BY entry_date DESC;

-- Test Query 4: Daily OHLC aggregate query
EXPLAIN ANALYZE
SELECT * FROM options_daily_ohlc
WHERE symbol = 'AAPL'
  AND bucket >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY bucket DESC;

-- ============================================================================
-- PART 6: MONITORING & MAINTENANCE
-- ============================================================================

-- Create view for index usage statistics
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    CASE
        WHEN idx_scan = 0 THEN 'UNUSED'
        WHEN idx_scan < 100 THEN 'LOW USAGE'
        ELSE 'ACTIVE'
    END AS usage_category
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;

COMMENT ON VIEW index_usage_stats IS
'Monitor index usage and identify unused indexes';

-- Create view for slow query monitoring
CREATE OR REPLACE VIEW slow_query_stats AS
SELECT
    query,
    calls,
    total_exec_time / 1000 AS total_time_sec,
    mean_exec_time / 1000 AS mean_time_sec,
    max_exec_time / 1000 AS max_time_sec,
    stddev_exec_time / 1000 AS stddev_time_sec,
    rows
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- Queries slower than 100ms on average
ORDER BY mean_exec_time DESC
LIMIT 20;

COMMENT ON VIEW slow_query_stats IS
'Identify slow queries requiring optimization (requires pg_stat_statements extension)';

-- Create view for TimescaleDB chunk information
CREATE OR REPLACE VIEW hypertable_chunks_info AS
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    pg_size_pretty(pg_total_relation_size(chunk_schema || '.' || chunk_name)) AS total_size,
    is_compressed
FROM timescaledb_information.chunks
ORDER BY range_start DESC;

COMMENT ON VIEW hypertable_chunks_info IS
'Monitor TimescaleDB chunks';

-- ============================================================================
-- PART 7: AUTOMATED MAINTENANCE JOBS
-- ============================================================================

-- Create function for automated vacuum and analyze
CREATE OR REPLACE FUNCTION maintain_database()
RETURNS void AS $$
BEGIN
    -- Vacuum and analyze main tables
    VACUUM ANALYZE options_prices;
    VACUUM ANALYZE positions;
    VACUUM ANALYZE orders;
    VACUUM ANALYZE ml_models;
    VACUUM ANALYZE model_predictions;

    -- Refresh continuous aggregates manually if needed
    -- (Usually handled by automatic refresh policy)

    -- Log maintenance
    RAISE NOTICE 'Database maintenance completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION maintain_database() IS
'Automated database maintenance (vacuum, analyze, refresh)';

-- Schedule maintenance job (requires pg_cron extension)
-- SELECT cron.schedule('database-maintenance', '0 3 * * *', 'SELECT maintain_database()');

-- ============================================================================
-- COMPLETION SUMMARY
-- ============================================================================

-- Display optimization summary
DO $$
DECLARE
    index_count INT;
    aggregate_count INT;
BEGIN
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public'
      AND indexname LIKE 'idx_%';

    SELECT COUNT(*) INTO aggregate_count
    FROM pg_matviews
    WHERE schemaname = 'public';

    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATABASE OPTIMIZATION COMPLETE';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Indexes Created: %', index_count;
    RAISE NOTICE 'Continuous Aggregates: %', aggregate_count;
    RAISE NOTICE 'Compression Policy: Enabled (7-day threshold)';
    RAISE NOTICE 'Retention Policy: 2 years (options_prices)';
    RAISE NOTICE '';
    RAISE NOTICE 'Expected Performance Improvements:';
    RAISE NOTICE '  - Query Latency: 5-10x faster';
    RAISE NOTICE '  - Storage Usage: -80%% (compression)';
    RAISE NOTICE '  - Analytics Queries: 10-50x faster';
    RAISE NOTICE '';
    RAISE NOTICE 'Next Steps:';
    RAISE NOTICE '  1. Monitor query performance with EXPLAIN ANALYZE';
    RAISE NOTICE '  2. Check index usage: SELECT * FROM index_usage_stats;';
    RAISE NOTICE '  3. Monitor compression: SELECT * FROM hypertable_chunks_info;';
    RAISE NOTICE '  4. Review slow queries: SELECT * FROM slow_query_stats;';
    RAISE NOTICE '========================================';
    RAISE NOTICE '';
END $$;

-- ============================================================================
-- PART 8: ADDITIONAL STRATEGIC INDEXES (HIGH-CONCURRENCY OPTIMIZATIONS)
-- ============================================================================

-- Enable pg_stat_statements extension for query monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Cover index for portfolio dashboard (includes columns to avoid table lookup)
-- This index covers the most common dashboard query pattern
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_dashboard
ON portfolios (user_id, created_at DESC)
INCLUDE (name, cash_balance);

COMMENT ON INDEX idx_portfolios_user_dashboard IS
'Cover index for portfolio dashboard - avoids table lookup for common queries';

-- Cover index for position aggregation (closed positions P&L)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_portfolio_pnl
ON positions (portfolio_id, status)
INCLUDE (realized_pnl, entry_price, quantity)
WHERE status = 'closed';

COMMENT ON INDEX idx_positions_portfolio_pnl IS
'Cover index for closed position P&L aggregation';

-- BRIN index for time-series queries (much smaller than B-tree, good for time-ordered data)
-- BRIN indexes are 100-1000x smaller than B-tree for time-series data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_prices_time_brin
ON options_prices USING BRIN (time)
WITH (pages_per_range = 128);

COMMENT ON INDEX idx_options_prices_time_brin IS
'BRIN index for time-range queries - very compact for time-series data';

-- Partial index for active users only (reduces index size significantly)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active_tier
ON users (tier, created_at DESC)
WHERE is_active = true;

COMMENT ON INDEX idx_users_active_tier IS
'Partial index for active users - smaller and faster than full index';

-- Partial index for pending orders (most frequently queried status)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_pending_quick
ON orders (created_at DESC)
INCLUDE (user_id, portfolio_id, symbol, side, quantity, order_type)
WHERE status = 'pending';

COMMENT ON INDEX idx_orders_pending_quick IS
'Cover index for pending orders - includes common columns';

-- ============================================================================
-- PART 9: PORTFOLIO SUMMARY MATERIALIZED VIEW
-- ============================================================================
-- Pre-aggregated portfolio statistics for dashboard performance
-- Avoids expensive JOINs and aggregations on every request

DROP MATERIALIZED VIEW IF EXISTS portfolio_summary CASCADE;

CREATE MATERIALIZED VIEW portfolio_summary AS
SELECT
    p.id AS portfolio_id,
    p.user_id,
    p.name,
    p.cash_balance,
    p.created_at,
    COUNT(pos.id) FILTER (WHERE pos.status = 'open') AS open_positions_count,
    COUNT(pos.id) FILTER (WHERE pos.status = 'closed') AS closed_positions_count,
    COALESCE(SUM(pos.realized_pnl) FILTER (WHERE pos.status = 'closed'), 0) AS total_realized_pnl,
    COALESCE(SUM(pos.quantity * pos.entry_price) FILTER (WHERE pos.status = 'open'), 0) AS open_position_value,
    p.cash_balance + COALESCE(SUM(pos.quantity * pos.entry_price) FILTER (WHERE pos.status = 'open'), 0) AS total_portfolio_value,
    MAX(pos.entry_date) AS last_trade_date,
    COUNT(DISTINCT pos.symbol) FILTER (WHERE pos.status = 'open') AS unique_symbols
FROM portfolios p
LEFT JOIN positions pos ON pos.portfolio_id = p.id
GROUP BY p.id, p.user_id, p.name, p.cash_balance, p.created_at;

-- Unique index required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_summary_id
ON portfolio_summary (portfolio_id);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_portfolio_summary_user
ON portfolio_summary (user_id, total_portfolio_value DESC);

COMMENT ON MATERIALIZED VIEW portfolio_summary IS
'Pre-aggregated portfolio statistics - refresh every 5 minutes for dashboard performance';

-- Function to refresh portfolio summary (call from application or pg_cron)
CREATE OR REPLACE FUNCTION refresh_portfolio_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary;
    RAISE NOTICE 'Portfolio summary refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 10: QUERY ANALYSIS HELPERS
-- ============================================================================

-- View to identify slow queries
CREATE OR REPLACE VIEW query_analysis AS
SELECT
    queryid,
    LEFT(query, 100) AS query_preview,
    calls,
    ROUND(total_exec_time::numeric / 1000, 2) AS total_time_sec,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND(max_exec_time::numeric, 2) AS max_time_ms,
    rows,
    ROUND(100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_pct
FROM pg_stat_statements
WHERE calls > 10
ORDER BY mean_exec_time DESC
LIMIT 50;

COMMENT ON VIEW query_analysis IS
'Top 50 queries by average execution time (for optimization targeting)';

-- View to find missing indexes (sequential scans on large tables)
CREATE OR REPLACE VIEW missing_indexes AS
SELECT
    schemaname,
    relname AS tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    CASE
        WHEN seq_scan > 0
        THEN ROUND(100.0 * idx_scan / (seq_scan + idx_scan), 2)
        ELSE 100
    END AS idx_usage_pct,
    n_live_tup AS estimated_rows
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan
  AND n_live_tup > 10000
ORDER BY seq_tup_read DESC;

COMMENT ON VIEW missing_indexes IS
'Tables with more sequential scans than index scans (candidates for new indexes)';

-- Function to reset query statistics (call after optimization)
CREATE OR REPLACE FUNCTION reset_query_stats()
RETURNS void AS $$
BEGIN
    PERFORM pg_stat_statements_reset();
    RAISE NOTICE 'Query statistics reset at %', NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- PART 11: CONNECTION MONITORING
-- ============================================================================

-- View for connection pool monitoring
CREATE OR REPLACE VIEW connection_stats AS
SELECT
    datname AS database,
    usename AS username,
    application_name,
    client_addr,
    state,
    COUNT(*) AS connection_count,
    MAX(EXTRACT(EPOCH FROM (NOW() - backend_start))) AS max_connection_age_sec,
    MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) AS max_idle_time_sec
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY datname, usename, application_name, client_addr, state
ORDER BY connection_count DESC;

COMMENT ON VIEW connection_stats IS
'Monitor active database connections by application and state';

-- View for long-running queries
CREATE OR REPLACE VIEW long_running_queries AS
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    EXTRACT(EPOCH FROM (NOW() - query_start)) AS duration_sec,
    LEFT(query, 200) AS query_preview,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND query_start < NOW() - INTERVAL '30 seconds'
  AND datname = current_database()
ORDER BY query_start;

COMMENT ON VIEW long_running_queries IS
'Queries running longer than 30 seconds (potential issues)';

-- ============================================================================
-- FINAL SUMMARY
-- ============================================================================

DO $$
DECLARE
    total_indexes INT;
    cover_indexes INT;
    partial_indexes INT;
    brin_indexes INT;
BEGIN
    SELECT COUNT(*) INTO total_indexes
    FROM pg_indexes WHERE schemaname = 'public';

    SELECT COUNT(*) INTO cover_indexes
    FROM pg_indexes
    WHERE schemaname = 'public' AND indexdef LIKE '%INCLUDE%';

    SELECT COUNT(*) INTO partial_indexes
    FROM pg_indexes
    WHERE schemaname = 'public' AND indexdef LIKE '%WHERE%';

    SELECT COUNT(*) INTO brin_indexes
    FROM pg_indexes
    WHERE schemaname = 'public' AND indexdef LIKE '%USING brin%';

    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'ADDITIONAL OPTIMIZATIONS APPLIED';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Total Indexes: %', total_indexes;
    RAISE NOTICE 'Cover Indexes (INCLUDE): %', cover_indexes;
    RAISE NOTICE 'Partial Indexes (WHERE): %', partial_indexes;
    RAISE NOTICE 'BRIN Indexes: %', brin_indexes;
    RAISE NOTICE '';
    RAISE NOTICE 'New Features:';
    RAISE NOTICE '  - portfolio_summary materialized view';
    RAISE NOTICE '  - query_analysis view for slow query detection';
    RAISE NOTICE '  - missing_indexes view for optimization hints';
    RAISE NOTICE '  - connection_stats view for pool monitoring';
    RAISE NOTICE '  - long_running_queries view for issue detection';
    RAISE NOTICE '';
    RAISE NOTICE 'Maintenance Functions:';
    RAISE NOTICE '  - refresh_portfolio_summary()';
    RAISE NOTICE '  - reset_query_stats()';
    RAISE NOTICE '  - maintain_database()';
    RAISE NOTICE '========================================';
END $$;


-- ============================================================================
-- PART 12: CACHE INVALIDATION TRIGGERS
-- ============================================================================
-- PostgreSQL NOTIFY triggers for real-time cache invalidation
-- The application listens on these channels to invalidate Redis cache entries
-- ============================================================================

-- Create notify function for user changes
CREATE OR REPLACE FUNCTION notify_user_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'users',
                'operation', TG_OP,
                'id', OLD.id::text,
                'tags', ARRAY['user:' || OLD.id::text]
            )::text
        );
        RETURN OLD;
    ELSE
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'users',
                'operation', TG_OP,
                'id', NEW.id::text,
                'tags', ARRAY['user:' || NEW.id::text]
            )::text
        );
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create notify function for portfolio changes
CREATE OR REPLACE FUNCTION notify_portfolio_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'portfolios',
                'operation', TG_OP,
                'id', OLD.id::text,
                'user_id', OLD.user_id::text,
                'tags', ARRAY[
                    'portfolio:' || OLD.id::text,
                    'user:' || OLD.user_id::text
                ]
            )::text
        );
        RETURN OLD;
    ELSE
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'portfolios',
                'operation', TG_OP,
                'id', NEW.id::text,
                'user_id', NEW.user_id::text,
                'tags', ARRAY[
                    'portfolio:' || NEW.id::text,
                    'user:' || NEW.user_id::text
                ]
            )::text
        );
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create notify function for position changes
CREATE OR REPLACE FUNCTION notify_position_change()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id UUID;
BEGIN
    -- Get user_id from portfolio
    IF TG_OP = 'DELETE' THEN
        SELECT user_id INTO v_user_id FROM portfolios WHERE id = OLD.portfolio_id;
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'positions',
                'operation', TG_OP,
                'id', OLD.id::text,
                'portfolio_id', OLD.portfolio_id::text,
                'user_id', v_user_id::text,
                'tags', ARRAY[
                    'position:' || OLD.id::text,
                    'portfolio:' || OLD.portfolio_id::text,
                    'user:' || v_user_id::text,
                    'position:list'
                ]
            )::text
        );
        RETURN OLD;
    ELSE
        SELECT user_id INTO v_user_id FROM portfolios WHERE id = NEW.portfolio_id;
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'positions',
                'operation', TG_OP,
                'id', NEW.id::text,
                'portfolio_id', NEW.portfolio_id::text,
                'user_id', v_user_id::text,
                'tags', ARRAY[
                    'position:' || NEW.id::text,
                    'portfolio:' || NEW.portfolio_id::text,
                    'user:' || v_user_id::text,
                    'position:list'
                ]
            )::text
        );
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create notify function for order changes
CREATE OR REPLACE FUNCTION notify_order_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'orders',
                'operation', TG_OP,
                'id', OLD.id::text,
                'user_id', OLD.user_id::text,
                'portfolio_id', OLD.portfolio_id::text,
                'tags', ARRAY[
                    'order:' || OLD.id::text,
                    'portfolio:' || OLD.portfolio_id::text,
                    'user:' || OLD.user_id::text
                ]
            )::text
        );
        RETURN OLD;
    ELSE
        PERFORM pg_notify('cache_invalidation',
            json_build_object(
                'table', 'orders',
                'operation', TG_OP,
                'id', NEW.id::text,
                'user_id', NEW.user_id::text,
                'portfolio_id', NEW.portfolio_id::text,
                'status', NEW.status,
                'tags', ARRAY[
                    'order:' || NEW.id::text,
                    'portfolio:' || NEW.portfolio_id::text,
                    'user:' || NEW.user_id::text
                ]
            )::text
        );
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to tables
DROP TRIGGER IF EXISTS trg_user_cache_invalidation ON users;
CREATE TRIGGER trg_user_cache_invalidation
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION notify_user_change();

DROP TRIGGER IF EXISTS trg_portfolio_cache_invalidation ON portfolios;
CREATE TRIGGER trg_portfolio_cache_invalidation
    AFTER INSERT OR UPDATE OR DELETE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION notify_portfolio_change();

DROP TRIGGER IF EXISTS trg_position_cache_invalidation ON positions;
CREATE TRIGGER trg_position_cache_invalidation
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW EXECUTE FUNCTION notify_position_change();

DROP TRIGGER IF EXISTS trg_order_cache_invalidation ON orders;
CREATE TRIGGER trg_order_cache_invalidation
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION notify_order_change();

COMMENT ON FUNCTION notify_user_change() IS 'Sends NOTIFY for user cache invalidation';
COMMENT ON FUNCTION notify_portfolio_change() IS 'Sends NOTIFY for portfolio cache invalidation';
COMMENT ON FUNCTION notify_position_change() IS 'Sends NOTIFY for position cache invalidation';
COMMENT ON FUNCTION notify_order_change() IS 'Sends NOTIFY for order cache invalidation';

-- ============================================================================
-- PART 13: CACHE LISTENER HELPER
-- ============================================================================
-- This function can be called to see recent cache invalidation events
-- Useful for debugging and monitoring

CREATE OR REPLACE VIEW recent_cache_invalidations AS
SELECT
    'To see real-time cache invalidations, run:' AS instructions,
    'LISTEN cache_invalidation;' AS command,
    'Then perform INSERT/UPDATE/DELETE on users, portfolios, positions, or orders' AS note;

COMMENT ON VIEW recent_cache_invalidations IS 'Instructions for monitoring cache invalidation events';

-- Final summary
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'CACHE INVALIDATION TRIGGERS CREATED';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Triggers installed on:';
    RAISE NOTICE '  - users (trg_user_cache_invalidation)';
    RAISE NOTICE '  - portfolios (trg_portfolio_cache_invalidation)';
    RAISE NOTICE '  - positions (trg_position_cache_invalidation)';
    RAISE NOTICE '  - orders (trg_order_cache_invalidation)';
    RAISE NOTICE '';
    RAISE NOTICE 'Channel: cache_invalidation';
    RAISE NOTICE 'Payload: JSON with table, operation, id, tags';
    RAISE NOTICE '';
    RAISE NOTICE 'Application should:';
    RAISE NOTICE '  1. LISTEN cache_invalidation';
    RAISE NOTICE '  2. Parse JSON payload';
    RAISE NOTICE '  3. Invalidate Redis keys by tags';
    RAISE NOTICE '========================================';
END $$;
