-- ============================================================================
-- Black-Scholes Option Pricing Platform - Consolidated Core Schema
-- ============================================================================

-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;
CREATE EXTENSION IF NOT EXISTS "vector";

-- 2. Custom Types for Optimization
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_tier') THEN
        CREATE TYPE user_tier AS ENUM ('free', 'pro', 'enterprise');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_side') THEN
        CREATE TYPE order_side AS ENUM ('buy', 'sell');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_status') THEN
        CREATE TYPE order_status AS ENUM ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_type') THEN
        CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'position_status') THEN
        CREATE TYPE position_status AS ENUM ('open', 'closed');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ml_algorithm') THEN
        CREATE TYPE ml_algorithm AS ENUM ('xgboost', 'lightgbm', 'neural_network', 'random_forest', 'svm', 'ensemble');
    END IF;
END $$;

-- 3. Core Tables (Aligned with Better-Auth)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255),
    full_name VARCHAR(255),
    tier user_tier DEFAULT 'free',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    mfa_backup_codes TEXT
) WITH (FILLFACTOR = 90);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    ip_address VARCHAR(50),
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
) WITH (FILLFACTOR = 90);

CREATE TABLE IF NOT EXISTS oauth_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, provider_id)
);

CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS oauth2_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id VARCHAR(100) UNIQUE NOT NULL,
    client_secret VARCHAR(255) NOT NULL,
    redirect_uris TEXT[],
    scopes TEXT[],
    is_confidential BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE
);

-- 4. Options Pricing & Market Data (Hypertables defined in 02-hypertables.sql)
CREATE TABLE IF NOT EXISTS options_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    strike NUMERIC(12, 2) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    bid NUMERIC(12, 4),
    ask NUMERIC(12, 4),
    last NUMERIC(12, 4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    rho DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, strike, expiry, option_type)
) WITH (FILLFACTOR = 100);

ALTER TABLE options_prices ALTER COLUMN symbol SET STATISTICS 500;

CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC(15, 4) NOT NULL,
    volume INTEGER,
    side order_side -- Reference order_side enum
) WITH (FILLFACTOR = 100);

-- 5. Portfolios, Positions & Orders
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    cash_balance NUMERIC(15, 2) DEFAULT 0.00 CHECK (cash_balance >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
) WITH (FILLFACTOR = 90);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    strike NUMERIC(12, 2),
    expiry DATE,
    option_type VARCHAR(4) CHECK (option_type IN ('call', 'put')),
    quantity INTEGER NOT NULL CHECK (quantity != 0),
    entry_price NUMERIC(12, 4) NOT NULL,
    entry_date TIMESTAMPTZ DEFAULT NOW(),
    current_price NUMERIC(12, 4),
    exit_price NUMERIC(12, 4),
    exit_date TIMESTAMPTZ,
    realized_pnl NUMERIC(15, 2),
    status position_status DEFAULT 'open',
    CONSTRAINT exit_price_requires_exit_date CHECK (
        (exit_price IS NULL AND exit_date IS NULL) OR
        (exit_price IS NOT NULL AND exit_date IS NOT NULL)
    )
) WITH (FILLFACTOR = 90);

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    strike NUMERIC(12, 2),
    expiry DATE,
    option_type VARCHAR(4) CHECK (option_type IN ('call', 'put')),
    side order_side NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    order_type order_type NOT NULL,
    limit_price NUMERIC(12, 4),
    stop_price NUMERIC(12, 4),
    status order_status DEFAULT 'pending',
    filled_quantity INTEGER DEFAULT 0,
    filled_price NUMERIC(12, 4),
    broker VARCHAR(50),
    broker_order_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT limit_order_requires_limit_price CHECK (
        (order_type != 'limit' AND order_type != 'stop_limit') OR limit_price IS NOT NULL
    ),
    CONSTRAINT stop_order_requires_stop_price CHECK (
        (order_type != 'stop' AND order_type != 'stop_limit') OR stop_price IS NOT NULL
    )
) WITH (FILLFACTOR = 90);

-- 6. ML Models & Predictions
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    algorithm ml_algorithm NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    hyperparameters JSONB,
    training_metrics JSONB,
    model_artifact_url VARCHAR(500),
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_production BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    input_features JSONB NOT NULL,
    predicted_price NUMERIC(12, 4) NOT NULL,
    actual_price NUMERIC(12, 4),
    prediction_error NUMERIC(12, 4),
    actual_value NUMERIC,
    PRIMARY KEY (id, timestamp)
) WITH (FILLFACTOR = 100);

-- 7. Utility Tables
CREATE UNLOGGED TABLE IF NOT EXISTS rate_limits (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(100) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, endpoint, window_start)
);

CREATE TABLE IF NOT EXISTS request_logs (
    created_at TIMESTAMPTZ NOT NULL, 
    status_code SMALLINT, 
    path TEXT, 
    method VARCHAR(10), 
    duration_ms DOUBLE PRECISION
) WITH (FILLFACTOR = 100);

-- 8. Functions & Triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_portfolios_updated_at ON portfolios;
CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_oauth_accounts_updated_at ON oauth_accounts;
CREATE TRIGGER update_oauth_accounts_updated_at BEFORE UPDATE ON oauth_accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
