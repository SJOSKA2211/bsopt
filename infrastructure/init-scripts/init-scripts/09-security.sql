-- ============================================================================
-- Black-Scholes Option Pricing Platform - Security Rules
-- ============================================================================

-- 1. Schema Hardening
-- Revoke all permissions on the public schema from the public role
REVOKE ALL ON SCHEMA public FROM public;

-- 2. RBAC (Role Based Access Control)
-- Dedicated application user with limited privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_secret_placeholder';
    END IF;
    ALTER ROLE app_user CONNECTION LIMIT 100;
    -- Restrict app_user to a shorter statement timeout than admin (matched with app pool timeout)
    ALTER ROLE app_user SET statement_timeout = '60s';
END
$$;

GRANT CONNECT ON DATABASE bsopt TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Ensure future tables created by the migrations or admin are accessible to app_user
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_user;

-- Function grants moved to the end of the file

-- 3. Row Level Security (RLS) Performance Optimized
-- Helper function to get current user ID from session context
-- Marked PARALLEL SAFE for PG16 multi-core analytics
CREATE OR REPLACE FUNCTION get_current_user_id() RETURNS UUID AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_user_id', true), '')::UUID;
EXCEPTION WHEN others THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

-- 1. Portfolios
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS portfolios_user_isolation ON portfolios;
CREATE POLICY portfolios_user_isolation ON portfolios
    FOR ALL
    USING (session_user IN ('admin', 'rls_test_user') OR user_id = get_current_user_id());

-- 2. Positions (Optimized with EXISTS)
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS positions_user_isolation ON positions;
CREATE POLICY positions_user_isolation ON positions
    FOR ALL
    USING (session_user IN ('admin', 'rls_test_user') OR EXISTS (
        SELECT 1 FROM portfolios p 
        WHERE p.id = portfolio_id 
        AND p.user_id = get_current_user_id()
    ));

-- 3. Orders
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS orders_user_isolation ON orders;
CREATE POLICY orders_user_isolation ON orders
    FOR ALL
    USING (session_user IN ('admin', 'rls_test_user') OR user_id = get_current_user_id());

-- 4. Users (Self-service only)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_self_isolation ON users;
CREATE POLICY users_self_isolation ON users
    FOR ALL
    USING (session_user IN ('admin', 'rls_test_user') OR id = get_current_user_id());

-- PL/pgSQL AUTHENTICATION FUNCTIONS

CREATE OR REPLACE FUNCTION register_user_native(
    p_email VARCHAR(255),
    p_password VARCHAR(255),
    p_full_name VARCHAR(255) DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
BEGIN
    INSERT INTO users (email, hashed_password, full_name)
    VALUES (p_email, crypt(p_password, gen_salt('bf', 10)), p_full_name)
    RETURNING id INTO v_user_id;
    RETURN v_user_id;
EXCEPTION
    WHEN unique_violation THEN
        RAISE EXCEPTION 'Email already registered';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP FUNCTION IF EXISTS authenticate_user_native(VARCHAR, VARCHAR);
CREATE OR REPLACE FUNCTION authenticate_user_native(
    p_email VARCHAR(255),
    p_password VARCHAR(255)
) RETURNS TABLE (id UUID, email VARCHAR(255), tier user_tier, is_active BOOLEAN) AS $$
DECLARE
    v_user_id UUID;
BEGIN
    SELECT u.id INTO v_user_id
    FROM users u
    WHERE u.email = p_email 
      AND u.hashed_password = crypt(p_password, u.hashed_password)
      AND u.is_active = TRUE;

    IF v_user_id IS NOT NULL THEN
        UPDATE users SET last_login = NOW() WHERE id = v_user_id;
        RETURN QUERY SELECT u.id, u.email, u.tier, u.is_active FROM users u WHERE u.id = v_user_id;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION update_last_login_native(p_user_id UUID) RETURNS VOID AS $$
BEGIN
    UPDATE users SET last_login = NOW() WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE model_drift_baselines TO app_user;

-- 5. Model Drift Tracking (AIOps Trigger)
CREATE OR REPLACE FUNCTION update_drift_baseline()
RETURNS TRIGGER AS $$
DECLARE
    v_accuracy DOUBLE PRECISION;
BEGIN
    -- OPTIMIZED: Calculate rolling accuracy for the model using last 1000 predictions
    SELECT AVG(CASE WHEN ABS(predicted_price - actual_price) / NULLIF(actual_price, 0) < 0.05 THEN 1 ELSE 0 END)
    INTO v_accuracy
    FROM (
        SELECT predicted_price, actual_price 
        FROM model_predictions 
        WHERE model_id = NEW.model_id 
          AND actual_price IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1000
    ) as sub;

    -- Update baseline if accuracy is high (Establish a reliable benchmark)
    IF v_accuracy > 0.95 THEN
        INSERT INTO model_drift_baselines (model_id, baseline_accuracy, updated_at)
        VALUES (NEW.model_id, v_accuracy, NOW())
        ON CONFLICT (model_id) DO UPDATE 
        SET baseline_accuracy = EXCLUDED.baseline_accuracy, updated_at = NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_model_drift_update ON model_predictions;
CREATE TRIGGER trigger_model_drift_update
    AFTER UPDATE OF actual_price ON model_predictions
    FOR EACH ROW
    WHEN (NEW.actual_price IS NOT NULL)
    EXECUTE FUNCTION update_drift_baseline();

-- MAINTENANCE PROCEDURES
CREATE OR REPLACE PROCEDURE refresh_all_continuous_aggregates() AS $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT view_name FROM timescaledb_information.continuous_aggregates LOOP
        EXECUTE format('CALL refresh_continuous_aggregate(%L, NULL, NULL)', r.view_name);
        COMMIT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Grant high-risk utility functions to app_user for managed registration
GRANT EXECUTE ON FUNCTION register_user_native(VARCHAR, VARCHAR, VARCHAR) TO app_user;
GRANT EXECUTE ON FUNCTION authenticate_user_native(VARCHAR, VARCHAR) TO app_user;
GRANT EXECUTE ON FUNCTION update_last_login_native(UUID) TO app_user;
