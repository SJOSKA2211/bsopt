-- ============================================================================
-- Black-Scholes Option Pricing Platform - Standard Materialized Views
-- ============================================================================

-- 1. Market Stats (Leveraging Continuous Aggregates for performance)
-- Note: daily_stats_chained_cagg already provides OHLCV
DROP MATERIALIZED VIEW IF EXISTS market_stats_mv;
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
GROUP BY p.user_id, p.id, p.name, p.cash_balance;

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
GROUP BY user_id;

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
GROUP BY model_id, DATE_TRUNC('hour', timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_drift_metrics_id_hour ON model_drift_metrics_mv(model_id, window_hour);

-- 5. Latest Volatility Surface (Feed for Heston/Pricing models)
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
ORDER BY symbol, expiry, strike, option_type, time DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_vol_surface_unique 
ON latest_vol_surface (symbol, expiry, strike, option_type);
