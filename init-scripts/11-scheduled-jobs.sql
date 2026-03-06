-- ============================================================================
-- Black-Scholes Option Pricing Platform - Scheduled Jobs (TimescaleDB)
-- ============================================================================

-- 1. Procedure to refresh standard Materialized Views concurrently
-- This avoids blocking reads during the refresh process.
CREATE OR REPLACE PROCEDURE refresh_standard_materialized_views(job_id int, config jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE NOTICE 'Refreshing standard materialized views...';
    
    -- Refresh portfolio summary (High priority, but can be slightly stale)
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary_mv;
    
    -- Refresh trading stats
    REFRESH MATERIALIZED VIEW CONCURRENTLY trading_stats_mv;
    
    -- Refresh model drift metrics (Last 24h window)
    REFRESH MATERIALIZED VIEW CONCURRENTLY model_drift_metrics_mv;
    
    -- Refresh latest volatility surface (Heaviest refresh)
    REFRESH MATERIALIZED VIEW CONCURRENTLY latest_vol_surface;
    
    RAISE NOTICE 'Standard materialized views refreshed successfully.';
END;
$$;

-- 2. Schedule the refresh job
-- We refresh these every 5 minutes. This provides a good balance between
-- real-time accuracy and database load.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'refresh_standard_materialized_views') THEN
        PERFORM add_job(
            'refresh_standard_materialized_views'::regproc,
            '5 minutes'::interval,
            initial_start => (NOW() + INTERVAL '1 minute')::timestamptz
        );
    END IF;
END $$;

-- 3. Maintenance Job: Re-analyze tables with high churn
-- This helps the query planner stay accurate for tables that change frequently
-- but might not trigger autovacuum analyze quickly enough.
CREATE OR REPLACE PROCEDURE maintenance_reanalyze_churn_tables(job_id int, config jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    ANALYZE orders;
    ANALYZE positions;
    ANALYZE portfolios;
    -- Rate limits is UNLOGGED, still benefits from fresh stats
    ANALYZE rate_limits;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'maintenance_reanalyze_churn_tables') THEN
        PERFORM add_job(
            'maintenance_reanalyze_churn_tables'::regproc,
            '1 hour'::interval,
            initial_start => (NOW() + INTERVAL '30 minutes')::timestamptz
        );
    END IF;
END $$;

-- 4. Statistics: Reset pg_stat_statements weekly to keep it focused on recent patterns
-- This can be useful for identifying new bottlenecks without old noise.
CREATE OR REPLACE PROCEDURE weekly_stats_reset(job_id int, config jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_stat_statements_reset();
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'weekly_stats_reset') THEN
        PERFORM add_job(
            'weekly_stats_reset'::regproc,
            '7 days'::interval,
            initial_start => (date_trunc('week', NOW()) + INTERVAL '1 week')::timestamptz
        );
    END IF;
END $$;
