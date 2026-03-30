-- ============================================================================
-- Black-Scholes Option Pricing Platform - Audit Logs
-- ============================================================================

-- OPTIMIZED: Audit log for API/Request level tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    time TIMESTAMPTZ NOT NULL,
    method VARCHAR(10) NOT NULL,
    path TEXT NOT NULL,
    status_code SMALLINT NOT NULL,
    user_id UUID,
    client_ip INET NOT NULL,
    user_agent TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
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
    changed_data JSONB, -- Only store what changed for updates
    full_row JSONB,     -- Full row for inserts/deletes
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
    -- Skip INSERT auditing for high-volume tables to prevent space bloat
    IF (TG_OP = 'INSERT' AND TG_TABLE_NAME IN (
        'options_prices', 'market_ticks', 'audit_logs', 'request_logs', 
        'model_predictions', 'market_data_mesh', 'rate_limits'
    )) THEN
        RETURN NEW;
    END IF;

    -- Attempt to get current user ID
    BEGIN
        v_user_id := NULLIF(current_setting('app.current_user_id', true), '')::UUID;
    EXCEPTION WHEN OTHERS THEN
        v_user_id := NULL;
    END;

    IF (TG_OP = 'DELETE') THEN
        INSERT INTO data_audit_logs (table_name, operation, user_id, full_row, query)
        VALUES (TG_TABLE_NAME, TG_OP, v_user_id, row_to_json(OLD)::JSONB, current_query());
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        -- Calculate diff: ONLY changed fields
        SELECT jsonb_object_agg(n.key, n.value) INTO v_changed_fields
        FROM jsonb_each(row_to_json(NEW)::JSONB) n
        JOIN jsonb_each(row_to_json(OLD)::JSONB) o ON n.key = o.key
        WHERE n.value IS DISTINCT FROM o.value;

        -- Only log if something actually changed
        IF v_changed_fields IS NOT NULL THEN
            INSERT INTO data_audit_logs (table_name, operation, user_id, changed_data, full_row, query)
            VALUES (TG_TABLE_NAME, TG_OP, v_user_id, v_changed_fields, row_to_json(NEW)::JSONB, current_query());
        END IF;
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO data_audit_logs (table_name, operation, user_id, full_row, query)
        VALUES (TG_TABLE_NAME, TG_OP, v_user_id, row_to_json(NEW)::JSONB, current_query());
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ATTACH TRIGGERS TO CORE TABLES
DROP TRIGGER IF EXISTS audit_users ON users;
CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON users FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_portfolios ON portfolios;
CREATE TRIGGER audit_portfolios AFTER INSERT OR UPDATE OR DELETE ON portfolios FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_positions ON positions;
CREATE TRIGGER audit_positions AFTER INSERT OR UPDATE OR DELETE ON positions FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_orders ON orders;
CREATE TRIGGER audit_orders AFTER INSERT OR UPDATE OR DELETE ON orders FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
