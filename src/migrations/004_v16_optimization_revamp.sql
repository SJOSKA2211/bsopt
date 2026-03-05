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
