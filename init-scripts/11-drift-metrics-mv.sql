-- Materialized View for Model Drift Metrics
-- Pre-aggregates prediction errors and drift scores for the last 24 hours.

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

-- Function to refresh drift metrics
CREATE OR REPLACE FUNCTION refresh_model_drift_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY model_drift_metrics_mv;
END;
$$ LANGUAGE plpgsql;
