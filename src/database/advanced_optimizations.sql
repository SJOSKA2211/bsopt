-- ============================================================================
-- Black-Scholes Option Pricing Platform - ADVANCED Database Optimizations
-- ============================================================================

-- 1. Partitioning by Expiry (Declarative)
-- Note: Requires options_prices to be a partitioned table if not already.
-- For TimescaleDB, we use time-based partitioning on 'time', 
-- but we can add an index on 'expiry' or use sub-partitioning if supported.
-- Standard practice in Timescale is to use 'time' as the main dimension.
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_symbol 
ON options_prices (expiry, symbol, time DESC);

-- 2. Materialized View for Real-time Volatility Surface
-- This view provides the latest IV for each strike/expiry to feed the Heston FFT model.
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

-- 3. Advanced Continuous Aggregate for Drift Detection
-- Calculates the Z-score of latency and error rates for AIOps.
CREATE MATERIALIZED VIEW IF NOT EXISTS system_metric_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    AVG(latency_ms) as avg_latency,
    STDDEV(latency_ms) as std_latency,
    COUNT(*) as request_count,
    SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as error_count
FROM audit_logs
GROUP BY bucket;

-- 4. Automatic Drift Baseline Trigger
-- Automatically updates a baseline table when model performance drops significantly.
CREATE TABLE IF NOT EXISTS model_drift_baselines (
    model_id TEXT PRIMARY KEY,
    baseline_accuracy DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_drift_baseline()
RETURNS TRIGGER AS $$
DECLARE
    v_accuracy DOUBLE PRECISION;
BEGIN
    -- OPTIMIZED: Calculate rolling accuracy for the model
    SELECT AVG(CASE WHEN ABS(predicted_value - actual_value) / NULLIF(actual_value, 0) < 0.05 THEN 1 ELSE 0 END)
    INTO v_accuracy
    FROM predictions
    WHERE model_id = NEW.model_id;

    -- Update baseline if accuracy is high
    IF v_accuracy > 0.95 THEN
        INSERT INTO model_drift_baselines (model_id, baseline_accuracy, updated_at)
        VALUES (NEW.model_id, v_accuracy, NOW())
        ON CONFLICT (model_id) DO UPDATE 
        SET baseline_accuracy = EXCLUDED.baseline_accuracy, updated_at = NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. Hypertable for Market Mesh Ticks
-- If we have a market_ticks table, make it a hypertable
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);

-- 6. Compression for Market Mesh
ALTER TABLE market_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('market_ticks', INTERVAL '1 day');

-- 7. Data Retention Policies (OOM Protection)
-- Keep raw ticks for 7 days, options prices for 30 days.
SELECT add_retention_policy('market_ticks', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('options_prices', INTERVAL '30 days', if_not_exists => TRUE);

