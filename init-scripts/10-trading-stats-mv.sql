-- Materialized View for Trading Statistics
-- Pre-aggregates order counts and average fill prices per user.

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

-- Function to refresh trading stats
CREATE OR REPLACE FUNCTION refresh_trading_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY trading_stats_mv;
END;
$$ LANGUAGE plpgsql;
