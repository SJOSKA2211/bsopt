-- ============================================================================
-- Migration: V16 Optimization Revamp
-- Description: Applies structural optimizations from the PG16/TimescaleDB revamp
-- ============================================================================

-- 1. Optimize Greeks Data Types (Numeric -> Double Precision for speed)
ALTER TABLE options_prices 
    ALTER COLUMN implied_volatility TYPE DOUBLE PRECISION,
    ALTER COLUMN delta TYPE DOUBLE PRECISION,
    ALTER COLUMN gamma TYPE DOUBLE PRECISION,
    ALTER COLUMN vega TYPE DOUBLE PRECISION,
    ALTER COLUMN theta TYPE DOUBLE PRECISION,
    ALTER COLUMN rho TYPE DOUBLE PRECISION;

-- 2. Turn model_predictions into a Hypertable (if data exists, use migrate_data => true)
-- Note: This requires dropping the existing PK if it's not on timestamp
ALTER TABLE model_predictions DROP CONSTRAINT IF EXISTS model_predictions_pkey;
SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);

-- 3. Add missing columns to model_predictions
ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS symbol TEXT;

-- 4. Storage Tuning (FILLFACTOR)
ALTER TABLE users SET (FILLFACTOR = 90);
ALTER TABLE portfolios SET (FILLFACTOR = 90);
ALTER TABLE positions SET (FILLFACTOR = 90);
ALTER TABLE orders SET (FILLFACTOR = 90);
ALTER TABLE options_prices SET (FILLFACTOR = 100);
ALTER TABLE market_ticks SET (FILLFACTOR = 100);

-- 5. Enable Compression on Continuous Aggregates (Idempotent)
DO $$
BEGIN
    ALTER MATERIALIZED VIEW IF EXISTS minute_stats_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS hourly_stats_chained_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS daily_stats_chained_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS greeks_drift_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS iv_surface_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS market_ticks_1m_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS market_ticks_1h_cagg SET (timescaledb.compress = true);
    ALTER MATERIALIZED VIEW IF EXISTS market_mesh_daily_cagg SET (timescaledb.compress = true);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Some continuous aggregates could not be altered for compression';
END $$;

-- 6. RLS Hardening (Update policies)
-- (Handled by re-running 09-security.sql, but we include critical ones here)
ALTER TABLE portfolios FORCE ROW LEVEL SECURITY;
ALTER TABLE positions FORCE ROW LEVEL SECURITY;
ALTER TABLE orders FORCE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;

-- 7. Native ENUM Standard (v2.5)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'option_type') THEN
        CREATE TYPE option_type AS ENUM ('call', 'put');
    END IF;
END $$;

-- Convert existing columns to use the new ENUM (with explicit cast)
-- options_prices
ALTER TABLE options_prices 
    ALTER COLUMN option_type TYPE option_type USING option_type::option_type;

-- positions
ALTER TABLE positions 
    ALTER COLUMN option_type TYPE option_type USING option_type::option_type;

-- orders
ALTER TABLE orders 
    ALTER COLUMN option_type TYPE option_type USING option_type::option_type;

-- 8. Aggressive Autovacuum Tuning (v2.5)
ALTER TABLE users SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

ALTER TABLE sessions SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

ALTER TABLE positions SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);
