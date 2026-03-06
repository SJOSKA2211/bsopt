-- ============================================================================
-- Black-Scholes Option Pricing Platform - Index Optimization
-- ============================================================================

-- 1. High-Volume Hypertable Indexes (BRIN for time-ordered data)
-- Optimized for large-scale range scans with minimal storage overhead.
CREATE INDEX IF NOT EXISTS idx_options_prices_brin_time ON options_prices USING BRIN (time) WITH (pages_per_range = 32);
CREATE INDEX IF NOT EXISTS idx_market_ticks_brin_time ON market_ticks USING BRIN (time) WITH (pages_per_range = 16);

-- 2. JSONB GIN Indexes (Optimized for JSON path operations)
CREATE INDEX IF NOT EXISTS idx_ml_models_hyperparams_gin ON ml_models USING GIN (hyperparameters jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_ml_models_metrics_gin ON ml_models USING GIN (training_metrics jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_model_predictions_features_gin ON model_predictions USING GIN (input_features jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_audit_logs_metadata_gin ON audit_logs USING GIN (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_market_mesh_metadata_gin ON market_data_mesh USING GIN (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_mesh_market ON market_data_mesh (market);
CREATE INDEX IF NOT EXISTS idx_mesh_source_type ON market_data_mesh (source_type);

-- 3. Partial Indexes (Highly optimized for specific status filters)
CREATE INDEX IF NOT EXISTS idx_request_logs_errors ON request_logs (status_code, created_at DESC) WHERE status_code >= 400;
CREATE INDEX IF NOT EXISTS idx_orders_open ON orders (user_id, created_at DESC) WHERE status IN ('pending', 'partially_filled');
CREATE INDEX IF NOT EXISTS idx_positions_active ON positions (portfolio_id, symbol) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_users_active_pro ON users (tier) WHERE is_active = TRUE AND is_verified = TRUE;

-- 4. Advanced Composite Indexes (Enabling Index-Only Scans)
-- Use SP-GiST for 2D-like queries on time and strike (Better for sparse spatial data like options chains)
CREATE INDEX IF NOT EXISTS idx_options_prices_gist_strike_time ON options_prices USING GIST (strike, time);

-- Composite INCLUDE index for lightning-fast options chain lookups
-- Optimized for: symbol, expiry, strike, option_type + latest time
DROP INDEX IF EXISTS idx_options_prices_chain;
CREATE INDEX idx_options_prices_chain 
ON options_prices (symbol, expiry, strike, option_type, time DESC)
INCLUDE (bid, ask, last, implied_volatility, delta, gamma, vega, theta, rho);

-- Optimized market_ticks lookup with volume included for faster data points retrieval
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_price_time ON market_ticks (symbol, price, time DESC) INCLUDE (volume);

-- 5. Planner Statistics Optimization (Help query planner with high-cardinality financial data)
ALTER TABLE options_prices ALTER COLUMN symbol SET STATISTICS 1000;
ALTER TABLE options_prices ALTER COLUMN strike SET STATISTICS 1000;
ALTER TABLE market_ticks ALTER COLUMN symbol SET STATISTICS 1000;
ALTER TABLE model_predictions ALTER COLUMN model_id SET STATISTICS 500;

-- 6. Specialized Feature-Specific Indexes
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_only ON options_prices (expiry DESC);
CREATE INDEX IF NOT EXISTS idx_mesh_symbol_time ON market_data_mesh (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_time ON model_predictions(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_time ON model_predictions(model_id, timestamp DESC);

-- 7. Audit & Relationship Optimization
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_logs (user_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_path_time ON audit_logs (path, time DESC);
CREATE INDEX IF NOT EXISTS idx_oauth2_client_id ON oauth2_clients(client_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup ON rate_limits(user_id, endpoint, window_start);
