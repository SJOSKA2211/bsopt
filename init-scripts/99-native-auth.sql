-- Reconstructed Native Authentication Functions (High-Performance)
-- Used by api/routes/auth.py and src/database/crud.py

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION register_user_native(
    p_email VARCHAR,
    p_password VARCHAR,
    p_full_name VARCHAR
) RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
    v_password_hash VARCHAR;
BEGIN
    v_password_hash := crypt(p_password, gen_salt('bf'));
    INSERT INTO users (email, hashed_password, full_name, tier, is_active, is_verified, created_at)
    VALUES (p_email, v_password_hash, p_full_name, 'free'::user_tier, TRUE, FALSE, NOW())
    RETURNING id INTO v_user_id;
    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION authenticate_user_native(
    p_email VARCHAR,
    p_password VARCHAR
) RETURNS TABLE (
    id UUID,
    email VARCHAR,
    tier user_tier,
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
$$ LANGUAGE plpgsql;
