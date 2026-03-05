-- ============================================================================
-- Black-Scholes Option Pricing Platform - Audit Logs
-- ============================================================================

-- OPTIMIZED: Audit log for API/Request level tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    time TIMESTAMPTZ NOT NULL,
    method VARCHAR(10) NOT NULL,
    path TEXT NOT NULL,
    status_code SMALLINT NOT NULL,
    user_id UUID, -- Optimized to match users.id type
    client_ip INET NOT NULL,
    user_agent TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    metadata JSONB
) WITH (FILLFACTOR = 100);

SELECT create_hypertable('audit_logs', 'time', if_not_exists => TRUE);

-- ============================================================================
-- DATA AUDIT LOGS (Row-level tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_audit_logs (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_id UUID,
    old_data JSONB,
    new_data JSONB,
    query TEXT
);

SELECT create_hypertable('data_audit_logs', 'time', if_not_exists => TRUE);

-- ============================================================================
-- AUDIT TRIGGER FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id UUID;
    v_changed_fields JSONB;
BEGIN
    -- Skip INSERT auditing for high-volume time-series tables (options_prices, market_ticks, audit_logs)
    -- to prevent doubling storage footprint in 2GB environment.
    IF (TG_OP = 'INSERT' AND TG_TABLE_NAME IN ('options_prices', 'market_ticks', 'audit_logs', 'request_logs', 'model_predictions')) THEN
        RETURN NEW;
    END IF;

    -- Attempt to get the current user ID from the session setting
    BEGIN
        v_user_id := NULLIF(current_setting('app.current_user_id', true), '')::UUID;
    EXCEPTION WHEN OTHERS THEN
        v_user_id := NULL;
    END;

    IF (TG_OP = 'DELETE') THEN
        INSERT INTO data_audit_logs (table_name, operation, user_id, old_data, query)
        VALUES (TG_TABLE_NAME, TG_OP, v_user_id, row_to_json(OLD)::JSONB, current_query());
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        -- Storage Optimization: Calculate diff between NEW and OLD to store only changed fields
        SELECT jsonb_object_agg(n.key, n.value) INTO v_changed_fields
        FROM jsonb_each(row_to_json(NEW)::JSONB) n
        JOIN jsonb_each(row_to_json(OLD)::JSONB) o ON n.key = o.key
        WHERE n.value IS DISTINCT FROM o.value;

        INSERT INTO data_audit_logs (table_name, operation, user_id, old_data, new_data, query)
        VALUES (TG_TABLE_NAME, TG_OP, v_user_id, v_changed_fields, row_to_json(NEW)::JSONB, current_query());
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO data_audit_logs (table_name, operation, user_id, new_data, query)
        VALUES (TG_TABLE_NAME, TG_OP, v_user_id, row_to_json(NEW)::JSONB, current_query());
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
