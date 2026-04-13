-- ============================================================================
-- Black-Scholes Option Pricing Platform - Scheduled Jobs (TimescaleDB)
-- ============================================================================

-- 1. Procedure to refresh standard Materialized Views safely
-- This handles the case where CONCURRENTLY cannot be used (e.g. view not yet populated)
CREATE OR REPLACE PROCEDURE refresh_standard_materialized_views(job_id int, config jsonb)
LANGUAGE plpgsql
AS $$
DECLARE
    v_mv_name TEXT;
    v_mvs TEXT[] := ARRAY['portfolio_summary_mv', 'trading_stats_mv', 'model_drift_metrics_mv', 'latest_vol_surface'];
BEGIN
    RAISE NOTICE 'Starting standard materialized view refresh cycle...';
    
    FOREACH v_mv_name IN ARRAY v_mvs
    LOOP
        BEGIN
            -- Try concurrent refresh first
            EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I', v_mv_name);
            RAISE NOTICE 'Refreshed % CONCURRENTLY', v_mv_name;
        EXCEPTION WHEN OTHERS THEN
            -- Fallback to standard refresh if concurrent fails (e.g. view not populated)
            RAISE WARNING 'Concurrent refresh failed for %, falling back to standard refresh. Error: %', v_mv_name, SQLERRM;
            EXECUTE format('REFRESH MATERIALIZED VIEW %I', v_mv_name);
            RAISE NOTICE 'Refreshed % (Standard)', v_mv_name;
        END;
    END LOOP;
    
    RAISE NOTICE 'Standard materialized views refresh cycle completed successfully.';
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
CREATE OR REPLACE PROCEDURE maintenance_reanalyze_churn_tables(job_id int, config jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    ANALYZE orders;
    ANALYZE positions;
    ANALYZE portfolios;
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

-- 4. Statistics: Reset pg_stat_statements weekly
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

-- 5. pg_cron Scheduled Jobs: Native Partition Maintenance
CREATE OR REPLACE FUNCTION maintain_model_predictions_partitions()
RETURNS void AS $$
DECLARE
    next_month DATE := date_trunc('month', NOW() + INTERVAL '1 month');
    table_name TEXT := 'model_predictions_' || to_char(next_month, 'YYYY_MM');
    start_date TEXT := to_char(next_month, 'YYYY-MM-DD');
    end_date TEXT := to_char(next_month + INTERVAL '1 month', 'YYYY-MM-DD');
BEGIN
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF model_predictions FOR VALUES FROM (%L) TO (%L)', table_name, start_date, end_date);
    RAISE NOTICE 'Created partition % for model_predictions', table_name;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Run daily at 00:00 to ensure next month's partition exists
        PERFORM cron.schedule('maintain_predictions_partitions', '0 0 * * *', 'SELECT maintain_model_predictions_partitions();');
    ELSE
        RAISE WARNING 'pg_cron extension not found, skipping cron schedule.';
    END IF;
END $$;
