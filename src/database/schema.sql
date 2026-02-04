-- ============================================================================
-- Black-Scholes Option Pricing Platform - Database Schema (Neon Optimized)
-- ============================================================================
-- PostgreSQL 15+ Native Partitioning
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    tier VARCHAR(20) DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tier_active ON users(tier, is_active);

COMMENT ON TABLE users IS 'Core user entity. Tiers (free, pro, enterprise) determine API rate limits and model access.';

-- ============================================================================
-- OAUTH2_CLIENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS oauth2_clients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(100) UNIQUE NOT NULL,
    client_secret VARCHAR(255) NOT NULL,
    redirect_uris TEXT[],
    scopes TEXT[],
    is_confidential BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_oauth2_client_id ON oauth2_clients(client_id);

-- ============================================================================
-- OPTIONS_PRICES TABLE (Native Partitioning)
-- ============================================================================

CREATE TABLE IF NOT EXISTS options_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strike NUMERIC(12, 2) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    bid NUMERIC(12, 4),
    ask NUMERIC(12, 4),
    last NUMERIC(12, 4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC(8, 6),
    delta NUMERIC(8, 6),
    gamma NUMERIC(8, 6),
    vega NUMERIC(8, 6),
    theta NUMERIC(8, 6),
    rho NUMERIC(8, 6),
    PRIMARY KEY (time, symbol, strike, expiry, option_type)
) PARTITION BY RANGE (time);

-- Initial Partition (February 2026)
CREATE TABLE IF NOT EXISTS options_prices_y2026m02 PARTITION OF options_prices
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Indexes for options_prices
CREATE INDEX IF NOT EXISTS idx_options_prices_symbol_time ON options_prices(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_time ON options_prices(expiry, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_chain ON options_prices(symbol, expiry, option_type, strike, time DESC);

COMMENT ON TABLE options_prices IS 'High-velocity time-series market data. Native Postgres range partitioning on ''time''.';

-- ============================================================================
-- PORTFOLIOS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    cash_balance NUMERIC(15, 2) DEFAULT 0.00 CHECK (cash_balance >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_portfolios_user_name ON portfolios(user_id, name);

-- ============================================================================
-- POSITIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    strike NUMERIC(12, 2),
    expiry DATE,
    option_type VARCHAR(4) CHECK (option_type IS NULL OR option_type IN ('call', 'put')),
    quantity INTEGER NOT NULL CHECK (quantity != 0),
    entry_price NUMERIC(12, 4) NOT NULL,
    entry_date TIMESTAMPTZ DEFAULT NOW(),
    exit_price NUMERIC(12, 4),
    exit_date TIMESTAMPTZ,
    realized_pnl NUMERIC(15, 2),
    status VARCHAR(10) DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    CONSTRAINT exit_price_requires_exit_date CHECK (
        (exit_price IS NULL AND exit_date IS NULL) OR
        (exit_price IS NOT NULL AND exit_date IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status ON positions(portfolio_id, status);

-- ============================================================================
-- ORDERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    strike NUMERIC(12, 2),
    expiry DATE,
    option_type VARCHAR(4) CHECK (option_type IS NULL OR option_type IN ('call', 'put')),
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    order_type VARCHAR(15) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    limit_price NUMERIC(12, 4),
    stop_price NUMERIC(12, 4),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected')),
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
);

CREATE INDEX IF NOT EXISTS idx_orders_user_created ON orders(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, created_at DESC);

-- ============================================================================
-- ML_MODELS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    hyperparameters JSONB,
    training_metrics JSONB,
    model_artifact_url VARCHAR(500),
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_production BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_ml_models_production ON ml_models(name, is_production);

-- ============================================================================
-- MODEL_PREDICTIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml_models(id) ON DELETE SET NULL,
    input_features JSONB NOT NULL,
    predicted_price NUMERIC(12, 4) NOT NULL,
    actual_price NUMERIC(12, 4),
    prediction_error NUMERIC(12, 4),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- RATE_LIMITS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS rate_limits (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(100) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, endpoint, window_start)
);

-- ============================================================================
-- TRIGGER FOR AUTO-UPDATING updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();