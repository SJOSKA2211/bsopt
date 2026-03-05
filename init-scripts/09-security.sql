-- ============================================================================
-- Black-Scholes Option Pricing Platform - Security Rules
-- ============================================================================

-- 1. Schema Hardening
-- Revoke all permissions on the public schema from the public role
REVOKE ALL ON SCHEMA public FROM public;
GRANT USAGE ON SCHEMA public TO public; -- Allow usage, but not creation

-- 2. RBAC
-- Dedicated application user with limited privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_secret_placeholder';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE bsopt TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- 3. Row Level Security (RLS) Performance Optimized

-- Helper function to get current user ID from session context
CREATE OR REPLACE FUNCTION get_current_user_id() RETURNS UUID AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_user_id', true), '')::UUID;
EXCEPTION WHEN others THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

-- 1. Portfolios
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS portfolios_user_isolation ON portfolios;
CREATE POLICY portfolios_user_isolation ON portfolios
    FOR ALL
    USING (user_id = get_current_user_id());

-- 2. Positions (Optimized with EXISTS)
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS positions_user_isolation ON positions;
CREATE POLICY positions_user_isolation ON positions
    FOR ALL
    USING (EXISTS (
        SELECT 1 FROM portfolios p 
        WHERE p.id = portfolio_id 
        AND p.user_id = get_current_user_id()
    ));

-- 3. Orders
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS orders_user_isolation ON orders;
CREATE POLICY orders_user_isolation ON orders
    FOR ALL
    USING (user_id = get_current_user_id());

-- 4. Users (Self-service only)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_self_isolation ON users;
CREATE POLICY users_self_isolation ON users
    FOR ALL
    USING (id = get_current_user_id());

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

CREATE OR REPLACE FUNCTION authenticate_user_native(
    p_email VARCHAR(255),
    p_password VARCHAR(255)
) RETURNS TABLE (id UUID, email VARCHAR(255), tier VARCHAR(20), is_active BOOLEAN) AS $$
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
