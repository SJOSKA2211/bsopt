-- Migration: 002_oauth_procedures.sql

BEGIN;

-- === Function to Upsert OAuth User ===
CREATE OR REPLACE FUNCTION upsert_oauth_user(
    p_provider VARCHAR,
    p_provider_id VARCHAR,
    p_email VARCHAR,
    p_access_token TEXT,
    p_refresh_token TEXT,
    p_expires_at TIMESTAMP WITH TIME ZONE
) RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
    v_account_id UUID;
BEGIN
    -- Check if user exists by provider_id
    SELECT user_id INTO v_user_id
    FROM oauth_accounts
    WHERE provider = p_provider AND provider_id = p_provider_id;

    -- If not found, check by email
    IF v_user_id IS NULL THEN
        SELECT id INTO v_user_id
        FROM users
        WHERE email = p_email;
    END IF;

    -- If still not found, create new user
    IF v_user_id IS NULL THEN
        INSERT INTO users (email, is_verified)
        VALUES (p_email, TRUE) -- OAuth users are verified by provider
        RETURNING id INTO v_user_id;
    END IF;

    -- Upsert OAuth account link
    INSERT INTO oauth_accounts (user_id, provider, provider_id, access_token, refresh_token, expires_at)
    VALUES (v_user_id, p_provider, p_provider_id, p_access_token, p_refresh_token, p_expires_at)
    ON CONFLICT (provider, provider_id) DO UPDATE
    SET access_token = EXCLUDED.access_token,
        refresh_token = EXCLUDED.refresh_token,
        expires_at = EXCLUDED.expires_at,
        updated_at = NOW();

    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

COMMIT;
