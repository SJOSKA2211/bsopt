-- Materialized View for Market Statistics
-- Pre-aggregates daily OHLCV for faster trend analysis.

CREATE MATERIALIZED VIEW IF NOT EXISTS market_stats_mv AS
SELECT 
    symbol,
    DATE(time) as trade_date,
    MIN(last) as low,
    MAX(last) as high,
    (array_agg(last ORDER BY time ASC))[1] as open,
    (array_agg(last ORDER BY time DESC))[1] as close,
    AVG(last) as avg_price,
    SUM(volume) as total_volume
FROM options_prices
GROUP BY symbol, DATE(time);

CREATE UNIQUE INDEX IF NOT EXISTS idx_market_stats_symbol_date ON market_stats_mv(symbol, trade_date);

-- Function to refresh market stats
CREATE OR REPLACE FUNCTION refresh_market_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY market_stats_mv;
END;
$$ LANGUAGE plpgsql;
