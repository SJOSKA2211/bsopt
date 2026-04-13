-- ============================================================================
-- Black-Scholes Option Pricing Platform - Standard Materialized Views
-- ============================================================================

-- 1. Market Stats (Leveraging Continuous Aggregates for performance)
-- Note: daily_stats_chained_cagg already provides OHLCV
DROP MATERIALIZED VIEW IF EXISTS market_stats_mv;
DROP VIEW IF EXISTS market_stats_view;
CREATE VIEW market_stats_view AS 
SELECT symbol, day as trade_date, low, high, avg_price, volume as total_volume 
FROM daily_stats_chained_cagg;

-- 2. Portfolio Summary (Transactional data)
CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_summary_mv AS 
SELECT 
    p.user_id, 
    p.id as portfolio_id, 
    p.name as portfolio_name, 
    p.cash_balance, 
    COUNT(pos.id) as total_positions, 
    COUNT(pos.id) FILTER (WHERE pos.status = 'open') as open_positions, 
    SUM(pos.realized_pnl) as total_realized_pnl, 
    MAX(pos.entry_date) as last_activity 
FROM portfolios p 
LEFT JOIN positions pos ON p.id = pos.portfolio_id 
GROUP BY p.user_id, p.id, p.name, p.cash_balance
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_summary_id ON portfolio_summary_mv(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_summary_user ON portfolio_summary_mv(user_id);

-- 3. Trading Stats (Transactional data)
CREATE MATERIALIZED VIEW IF NOT EXISTS trading_stats_mv AS 
SELECT 
    user_id, 
    COUNT(id) as total_orders, 
    COUNT(id) FILTER (WHERE status = 'filled') as filled_orders, 
    COUNT(id) FILTER (WHERE status = 'cancelled') as cancelled_orders, 
    AVG(filled_price) FILTER (WHERE status = 'filled') as avg_fill_price 
FROM orders 
GROUP BY user_id
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_trading_stats_user_id ON trading_stats_mv(user_id);

-- 4. Model Drift Metrics (Short-term analysis)
CREATE MATERIALIZED VIEW IF NOT EXISTS model_drift_metrics_mv AS 
SELECT 
    model_id, 
    DATE_TRUNC('hour', timestamp) as window_hour, 
    AVG(ABS(predicted_price - actual_price)) as mae, 
    SQRT(AVG(POWER(predicted_price - actual_price, 2))) as rmse, 
    COUNT(*) as prediction_count 
FROM model_predictions 
WHERE actual_price IS NOT NULL 
  AND timestamp >= NOW() - INTERVAL '24 hours' 
GROUP BY model_id, DATE_TRUNC('hour', timestamp)
WITH NO DATA;
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_drift_metrics_id_hour ON model_drift_metrics_mv(model_id, window_hour);

-- 5. Latest Volatility Surface (Skip-Scan candidate for high-volume hypertables)
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_vol_surface
AS
SELECT DISTINCT ON (symbol, expiry, strike, option_type)
    symbol,
    expiry,
    strike,
    option_type,
    implied_volatility,
    last as price,
    time
FROM options_prices
ORDER BY symbol, expiry, strike, option_type, time DESC
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_vol_surface_unique 
ON latest_vol_surface (symbol, expiry, strike, option_type);

-- 6. Performance Diagnostics (Query Bottleneck Detection)
DROP VIEW IF EXISTS pg_stat_sluggish_queries;
CREATE OR REPLACE VIEW pg_stat_sluggish_queries AS
SELECT 
    (total_exec_time / 1000 / 60) as total_min, 
    (total_exec_time / calls) as avg_ms, 
    query 
FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 20;

-- 7. Index Size Diagnostic
DROP VIEW IF EXISTS pg_stat_index_sizes;
CREATE OR REPLACE VIEW pg_stat_index_sizes AS
SELECT
    schemaname, relname, indexrelname,
    pg_size_pretty(pg_relation_size(pg_index.indexrelid)) AS index_size
FROM pg_stat_user_indexes
JOIN pg_index ON pg_stat_user_indexes.indexrelid = pg_index.indexrelid
WHERE pg_relation_size(pg_index.indexrelid) > 1024 * 1024; -- Only check indexes > 1MB

-- 8. Wait Event Diagnostic (PG16)
-- Helps identify why queries are waiting (I/O, locks, CPU, etc.)
DROP VIEW IF EXISTS pg_stat_wait_events;
CREATE OR REPLACE VIEW pg_stat_wait_events AS
SELECT 
    wait_event_type, 
    wait_event, 
    count(*) as count 
FROM pg_stat_activity 
WHERE wait_event IS NOT NULL 
GROUP BY wait_event_type, wait_event 
ORDER BY count DESC;

-- 9. Table Bloat Estimation (Rough estimate)
CREATE OR REPLACE VIEW pg_stat_bloat_estimation AS
SELECT
    schemaname, relname, 
    n_dead_tup, 
    n_live_tup,
    ROUND(n_dead_tup::numeric / (n_live_tup + n_dead_tup + 1), 4) AS bloat_ratio
FROM pg_stat_all_tables
WHERE (n_live_tup + n_dead_tup) > 1000
ORDER BY bloat_ratio DESC;
