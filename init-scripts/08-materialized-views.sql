-- ============================================================================
-- Black-Scholes Option Pricing Platform - Standard Materialized Views
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS market_stats_mv AS 
SELECT symbol, DATE(time) as trade_date, MIN(last) as low, MAX(last) as high, (array_agg(last ORDER BY time ASC))[1] as open, (array_agg(last ORDER BY time DESC))[1] as close, AVG(last) as avg_price, SUM(volume) as total_volume 
FROM options_prices GROUP BY symbol, DATE(time);
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_stats_symbol_date ON market_stats_mv(symbol, trade_date);

CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_summary_mv AS 
SELECT p.user_id, p.id as portfolio_id, p.name as portfolio_name, p.cash_balance, COUNT(pos.id) as total_positions, COUNT(pos.id) FILTER (WHERE pos.status = 'open') as open_positions, SUM(pos.realized_pnl) as total_realized_pnl, MAX(pos.entry_date) as last_activity 
FROM portfolios p LEFT JOIN positions pos ON p.id = pos.portfolio_id GROUP BY p.user_id, p.id, p.name, p.cash_balance;
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_summary_id ON portfolio_summary_mv(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_summary_user ON portfolio_summary_mv(user_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS trading_stats_mv AS 
SELECT user_id, COUNT(id) as total_orders, COUNT(id) FILTER (WHERE status = 'filled') as filled_orders, COUNT(id) FILTER (WHERE status = 'cancelled') as cancelled_orders, AVG(filled_price) FILTER (WHERE status = 'filled') as avg_fill_price 
FROM orders GROUP BY user_id;
CREATE UNIQUE INDEX IF NOT EXISTS idx_trading_stats_user_id ON trading_stats_mv(user_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS model_drift_metrics_mv AS 
SELECT model_id, DATE_TRUNC('hour', timestamp) as window_hour, AVG(ABS(predicted_price - actual_price)) as mae, SQRT(AVG(POWER(predicted_price - actual_price, 2))) as rmse, COUNT(*) as prediction_count 
FROM model_predictions WHERE actual_price IS NOT NULL AND timestamp >= NOW() - INTERVAL '24 hours' GROUP BY model_id, DATE_TRUNC('hour', timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_drift_metrics_id_hour ON model_drift_metrics_mv(model_id, window_hour);
