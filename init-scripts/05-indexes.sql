-- ============================================================================
-- Black-Scholes Option Pricing Platform - Index Optimization
-- ============================================================================

-- 1. High-Volume Hypertable Indexes (BRIN for time-ordered data)
CREATE INDEX IF NOT EXISTS idx_options_prices_brin_time ON options_prices USING BRIN (time) WITH (pages_per_range = 32);
CREATE INDEX IF NOT EXISTS idx_market_ticks_brin_time ON market_ticks USING BRIN (time) WITH (pages_per_range = 16);

-- 2. JSONB GIN Indexes (Optimized for JSON path operations)
CREATE INDEX IF NOT EXISTS idx_ml_models_hyperparams_gin ON ml_models USING GIN (hyperparameters jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_ml_models_metrics_gin ON ml_models USING GIN (training_metrics jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_model_predictions_features_gin ON model_predictions USING GIN (input_features jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_audit_logs_metadata_gin ON audit_logs USING GIN (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_mesh_metadata_gin ON market_data_mesh USING GIN (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_mesh_market ON market_data_mesh (market);
CREATE INDEX IF NOT EXISTS idx_mesh_source_type ON market_data_mesh (source_type);

-- 3. Partial Indexes (Highly optimized for specific status filters)
CREATE INDEX IF NOT EXISTS idx_request_logs_errors ON request_logs (status_code, created_at DESC) WHERE status_code >= 400;
CREATE INDEX IF NOT EXISTS idx_orders_open ON orders (user_id, created_at DESC) WHERE status IN ('pending', 'partially_filled');
CREATE INDEX IF NOT EXISTS idx_positions_active ON positions (portfolio_id, symbol) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_users_active_pro ON users (tier) WHERE is_active = TRUE AND is_verified = TRUE;

-- 4. Advanced Indexes for Options Chain
-- Use GIST for 2D-like queries on time and strike if needed, but B-Tree is often faster for exact symbol + range time
CREATE INDEX IF NOT EXISTS idx_options_prices_gist_time_strike ON options_prices USING GIST (time, strike);

-- Composite INCLUDE index for lightning-fast options chain lookups (Index-Only Scans)
DROP INDEX IF EXISTS idx_options_prices_chain;
CREATE INDEX idx_options_prices_chain 
ON options_prices (symbol, expiry, strike, option_type)
INCLUDE (bid, ask, last, implied_volatility, delta, gamma, vega, theta, rho);

-- 5. Pattern-Specific Lookup Indexes
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_only ON options_prices (expiry DESC);
CREATE INDEX IF NOT EXISTS idx_mesh_symbol_time ON market_data_mesh (symbol, time DESC);

-- Specialized index for market_ticks with price ranges (Index-Only scan candidate)
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_price_time ON market_ticks (symbol, price, time DESC) INCLUDE (volume);

-- 6. Maintenance & Audit Indexes
CREATE INDEX IF NOT EXISTS idx_options_prices_symbol_time ON options_prices(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_time ON options_prices(expiry, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_time ON market_ticks(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_logs (user_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_path_time ON audit_logs (path, time DESC);

-- Specialized indexes for data audit tracking
CREATE INDEX IF NOT EXISTS idx_data_audit_table_time ON data_audit_logs (table_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_data_audit_user_time ON data_audit_logs (user_id, time DESC);
