-- Materialized View for Portfolio Summary
-- Pre-aggregates portfolio stats (total value, realized PnL, position count) 
-- to enable near-instant dashboard loading.

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

-- Function to refresh portfolio summary
CREATE OR REPLACE FUNCTION refresh_portfolio_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary_mv;
END;
$$ LANGUAGE plpgsql;

-- Trigger to refresh on position changes (optional, usually better on a schedule for performance)
-- For now, we will rely on a scheduled background task.
