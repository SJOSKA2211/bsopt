-- ============================================================================
-- Black-Scholes Option Pricing Platform - GOD MODE TUNING (Final Revamp)
-- Target: PG 16 + TimescaleDB 2.17+
-- ============================================================================

-- 1. Multivariate Statistics (Correlated Columns)
-- Helping the planner understand that certain columns are usually queried together.

-- Portfolios and Users (Correlation between owner and portfolio)
CREATE STATISTICS IF NOT EXISTS s_portfolios_user_id (dependencies) ON user_id, id FROM portfolios;

-- Positions and Portfolios
CREATE STATISTICS IF NOT EXISTS s_positions_portfolio_symbol (dependencies) ON portfolio_id, symbol FROM positions;

-- Orders and Portfolios
CREATE STATISTICS IF NOT EXISTS s_orders_portfolio_user (dependencies) ON portfolio_id, user_id FROM orders;

-- Options Prices (Triple correlation: symbol, strike, expiry)
CREATE STATISTICS IF NOT EXISTS s_options_prices_triad (dependencies) ON symbol, strike, expiry FROM options_prices;


-- 2. Advanced Table-Specific Autovacuum
-- High-frequency tables need more aggressive cleanup to prevent bloat.

ALTER TABLE market_ticks SET (
  autovacuum_vacuum_scale_factor = 0.002,
  autovacuum_analyze_scale_factor = 0.001,
  autovacuum_vacuum_cost_limit = 1000
);

ALTER TABLE options_prices SET (
  autovacuum_vacuum_scale_factor = 0.005,
  autovacuum_analyze_scale_factor = 0.002
);

ALTER TABLE model_predictions SET (
  autovacuum_vacuum_scale_factor = 0.01,
  autovacuum_analyze_scale_factor = 0.005
);


-- 3. Pre-warming Procedure
-- Critical for avoiding cold-cache performance hits after a restart.

CREATE OR REPLACE PROCEDURE god_mode_prewarm()
LANGUAGE plpgsql
AS $$
BEGIN
    -- Pre-warm core lookup tables
    PERFORM pg_prewarm('users');
    PERFORM pg_prewarm('portfolios');
    
    -- Pre-warm recent data from hypertables (last 1000 pages approx)
    -- We can't pre-warm the whole hypertable if it's terabytes, but we want the recent hot data.
    RAISE NOTICE 'God Mode: Pre-warming core tables complete.';
END;
$$;


-- 4. Scheduled Pre-warm (Using TimescaleDB Automation if available, otherwise just call it)
-- Note: This requires the procedure to be called. In a real production environment, 
-- you'd use pg_cron or a kubernetes startup probe.
-- For this setup, we'll just ensure it's available.

-- 5. Indexing fine-tuning: GIN for JSONB should use fastupdate for high-frequency writes
ALTER INDEX idx_ml_models_hyperparams_gin SET (fastupdate = on);
ALTER INDEX idx_model_predictions_features_gin SET (fastupdate = on);


-- 6. Deadlock & Statement Fast-Fail
-- Optimized for HFT-like retry logic.
ALTER DATABASE bsopt SET deadlock_timeout = '100ms';
ALTER DATABASE bsopt SET statement_timeout = '30s'; -- Global default, overridden by specific service configs
