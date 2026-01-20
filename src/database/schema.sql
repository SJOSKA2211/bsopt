-- ============================================================================
-- Black-Scholes Option Pricing Platform - Database Schema
-- ============================================================================
-- PostgreSQL 15 with TimescaleDB
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

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

COMMENT ON TABLE users IS 'User accounts with tiered access (free, pro, enterprise)';

-- ============================================================================
-- OPTIONS_PRICES TABLE (TimescaleDB Hypertable)
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
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);

-- Indexes for options_prices
CREATE INDEX IF NOT EXISTS idx_options_prices_symbol_time ON options_prices(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_time ON options_prices(expiry, time DESC);
CREATE INDEX IF NOT EXISTS idx_options_prices_chain ON options_prices(symbol, expiry, option_type, strike, time DESC);

COMMENT ON TABLE options_prices IS 'Time-series options market data (TimescaleDB hypertable)';

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

CREATE INDEX IF NOT EXISTS idx_portfolios_user_created ON portfolios(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_portfolios_user_name ON portfolios(user_id, name);

COMMENT ON TABLE portfolios IS 'User portfolios for position tracking';

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
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_positions_expiry_status ON positions(expiry, status);

COMMENT ON TABLE positions IS 'Individual option positions (quantity > 0 = long, < 0 = short)';

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
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_created ON orders(portfolio_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_broker_lookup ON orders(broker, broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status, created_at DESC);

COMMENT ON TABLE orders IS 'Trading order management (market, limit, stop, stop_limit)';

-- ============================================================================
-- ML_MODELS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL CHECK (algorithm IN ('xgboost', 'lightgbm', 'neural_network', 'random_forest', 'svm', 'ensemble')),
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
CREATE INDEX IF NOT EXISTS idx_ml_models_version ON ml_models(name, version DESC);
CREATE INDEX IF NOT EXISTS idx_ml_models_created_by ON ml_models(created_by);

COMMENT ON TABLE ml_models IS 'ML model registry with versioning';

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

CREATE INDEX IF NOT EXISTS idx_model_predictions_model_time ON model_predictions(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_pending ON model_predictions(timestamp DESC) WHERE actual_price IS NULL;

COMMENT ON TABLE model_predictions IS 'Prediction logs for ML model monitoring';

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

CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup ON rate_limits(user_id, endpoint, window_start);

COMMENT ON TABLE rate_limits IS 'API rate limiting tracking by user and endpoint';

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

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Show all created tables
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATABASE SCHEMA CREATION COMPLETE';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - users';
    RAISE NOTICE '  - options_prices (TimescaleDB hypertable)';
    RAISE NOTICE '  - portfolios';
    RAISE NOTICE '  - positions';
    RAISE NOTICE '  - orders';
    RAISE NOTICE '  - ml_models';
    RAISE NOTICE '  - model_predictions';
    RAISE NOTICE '  - rate_limits';
    RAISE NOTICE '========================================';
END $$;
