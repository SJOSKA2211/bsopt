-- ============================================================================
-- God-Mode Security: RLS Policies & PL/pgSQL Auth
-- ============================================================================

-- Enable pgcrypto for password hashing
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- 1. Portfolios
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS portfolios_user_isolation ON portfolios;
CREATE POLICY portfolios_user_isolation ON portfolios
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- 2. Positions
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS positions_user_isolation ON positions;
CREATE POLICY positions_user_isolation ON positions
    FOR ALL
    USING (portfolio_id IN (SELECT id FROM portfolios WHERE user_id = current_setting('app.current_user_id')::UUID));

-- 3. Orders
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS orders_user_isolation ON orders;
CREATE POLICY orders_user_isolation ON orders
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- 4. Users (Self-service only)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS users_self_isolation ON users;
CREATE POLICY users_self_isolation ON users
    FOR ALL
    USING (id = current_setting('app.current_user_id')::UUID);

-- ============================================================================
-- PL/pgSQL AUTHENTICATION FUNCTIONS
-- ============================================================================

-- Function to register a new user
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

-- Function to authenticate a user
CREATE OR REPLACE FUNCTION authenticate_user_native(
    p_email VARCHAR(255),
    p_password VARCHAR(255)
) RETURNS TABLE (
    id UUID,
    email VARCHAR(255),
    tier VARCHAR(20),
    is_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT u.id, u.email, u.tier, u.is_active
    FROM users u
    WHERE u.email = p_email 
      AND u.hashed_password = crypt(p_password, u.hashed_password)
      AND u.is_active = TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update last login
CREATE OR REPLACE FUNCTION update_last_login_native(p_user_id UUID) RETURNS VOID AS $$
BEGIN
    UPDATE users SET last_login = NOW() WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- COMPLETION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'God-Mode Security implemented: RLS enabled, Auth functions created.';
END $$;
