-- 🚀 GOD-MODE DATABASE PERFORMANCE TUNING
-- ============================================================================
-- SOTA Indexing Strategies for Option Pricing and High-Frequency Trading
-- ============================================================================

-- 1. BRIN (Block Range Index) for High-Frequency Hypertables
-- BRIN is exponentially more space-efficient than B-Tree for large, naturally ordered datasets.
-- It works by storing the min/max values for ranges of blocks.
CREATE INDEX IF NOT EXISTS idx_options_prices_brin_time 
ON options_prices USING BRIN (time) WITH (pages_per_range = 32);

CREATE INDEX IF NOT EXISTS idx_market_ticks_brin_time 
ON market_ticks USING BRIN (time) WITH (pages_per_range = 16);

-- 2. GIN (Generalized Inverted Index) for JSONB columns
-- Essential for fast attribute-based searching within unstructured data.
CREATE INDEX IF NOT EXISTS idx_ml_models_hyperparams_gin ON ml_models USING GIN (hyperparameters);
CREATE INDEX IF NOT EXISTS idx_ml_models_metrics_gin ON ml_models USING GIN (training_metrics);
CREATE INDEX IF NOT EXISTS idx_model_predictions_features_gin ON model_predictions USING GIN (input_features);
CREATE INDEX IF NOT EXISTS idx_audit_logs_metadata_gin ON audit_logs USING GIN (metadata);
-- CREATE INDEX IF NOT EXISTS idx_request_logs_params_gin ON request_logs USING GIN (query_params);

-- 3. Partial Indexes for High-Traffic Error Patterns
-- request_logs is likely huge. Indexing only 'interesting' status codes saves space and search time.
CREATE INDEX IF NOT EXISTS idx_request_logs_errors 
ON request_logs (status_code, created_at DESC) 
WHERE status_code >= 400;

-- 4. Advanced Composite Index for Option Chains (Re-optimized)
-- We use INCLUDE to allow Index-Only Scans for the most common pricing queries.
-- This avoids heap lookups entirely for these columns.
DROP INDEX IF EXISTS idx_options_prices_chain;
CREATE INDEX idx_options_prices_chain 
ON options_prices (symbol, expiry, strike, option_type)
INCLUDE (bid, ask, last, implied_volatility, delta, gamma, vega, theta, rho);

-- 5. Analyze to ensure query planner uses the new indices
ANALYZE options_prices;
ANALYZE market_ticks;
ANALYZE request_logs;
ANALYZE audit_logs;