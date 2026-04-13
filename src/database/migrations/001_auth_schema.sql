-- Migration: 001_auth_schema.sql
-- Description: Sets up the initial schema for native Postgres authentication.

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- === Users ===
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- === Sessions ===
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- === OAuth Accounts ===
CREATE TABLE IF NOT EXISTS oauth_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- 'google', 'github'
    provider_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider, provider_id)
);

-- === Email Verification Tokens ===
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- === Functions ===

-- Function to create a user and hash password securely
CREATE OR REPLACE FUNCTION create_user(
    p_email VARCHAR,
    p_password VARCHAR
) RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
    v_password_hash VARCHAR;
BEGIN
    v_password_hash := crypt(p_password, gen_salt('bf'));
    INSERT INTO users (email, password_hash)
    VALUES (p_email, v_password_hash)
    RETURNING id INTO v_user_id;
    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to verify password
CREATE OR REPLACE FUNCTION verify_password(
    p_email VARCHAR,
    p_password VARCHAR
) RETURNS BOOLEAN AS $$
DECLARE
    v_password_hash VARCHAR;
BEGIN
    SELECT password_hash INTO v_password_hash
    FROM users
    WHERE email = p_email;
    
    IF v_password_hash IS NULL THEN
        RETURN FALSE;
    END IF;
    
    RETURN (v_password_hash = crypt(p_password, v_password_hash));
END;
$$ LANGUAGE plpgsql;

COMMIT;
