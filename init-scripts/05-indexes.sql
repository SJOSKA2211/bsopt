-- ============================================================================
-- Black-Scholes Option Pricing Platform - Index Optimization
-- ============================================================================

-- BRIN indexes for high volume append-only hypertables
CREATE INDEX IF NOT EXISTS idx_options_prices_brin_time ON options_prices USING BRIN (time) WITH (pages_per_range = 32);
CREATE INDEX IF NOT EXISTS idx_market_ticks_brin_time ON market_ticks USING BRIN (time) WITH (pages_per_range = 16);

-- GIN indexes for JSONB with optimized path ops
CREATE INDEX IF NOT EXISTS idx_ml_models_hyperparams_gin ON ml_models USING GIN (hyperparameters jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_ml_models_metrics_gin ON ml_models USING GIN (training_metrics jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_model_predictions_features_gin ON model_predictions USING GIN (input_features jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_audit_logs_metadata_gin ON audit_logs USING GIN (metadata jsonb_path_ops);

-- Partial indexes for common error logging
CREATE INDEX IF NOT EXISTS idx_request_logs_errors ON request_logs (status_code, created_at DESC) WHERE status_code >= 400;

-- GiST index for efficient range queries on time and strike
CREATE INDEX IF NOT EXISTS idx_options_prices_gist_time_strike ON options_prices USING GIST (time, strike);

-- Composite indexes for options chain lookup to enable index-only scans
DROP INDEX IF EXISTS idx_options_prices_chain;
CREATE INDEX idx_options_prices_chain 
ON options_prices (symbol, expiry, strike, option_type)
INCLUDE (bid, ask, last, implied_volatility, delta, gamma, vega, theta, rho);

CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_only ON options_prices (expiry DESC);
CREATE INDEX IF NOT EXISTS idx_mesh_symbol_time ON market_data_mesh (symbol, time DESC);

-- Specialized index for market_ticks with price ranges
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_price_time ON market_ticks (symbol, price, time DESC);

-- Standard lookup indexes for querying patterns
CREATE INDEX IF NOT EXISTS idx_options_prices_symbol_time ON options_prices(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_time ON options_prices(expiry, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_time ON market_ticks(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_logs (user_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_path_time ON audit_logs (path, time DESC);
