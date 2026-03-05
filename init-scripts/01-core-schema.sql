-- ============================================================================
-- Black-Scholes Option Pricing Platform - Core Schema
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    tier VARCHAR(20) DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    reset_token_expires_at TIMESTAMPTZ,
    is_mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    mfa_backup_codes TEXT
) WITH (FILLFACTOR = 90);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
) WITH (FILLFACTOR = 90);

-- AUDIT: Track user changes
DROP TRIGGER IF EXISTS audit_users ON users;
CREATE TRIGGER audit_users
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TABLE IF NOT EXISTS oauth_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
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

CREATE INDEX IF NOT EXISTS idx_oauth2_client_id ON oauth2_clients(client_id);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tier_active ON users(tier, is_active);

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
    implied_volatility DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    rho DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, strike, expiry, option_type)
) WITH (FILLFACTOR = 100);

-- Increase statistics target for symbol column for better query planning in large datasets
ALTER TABLE options_prices ALTER COLUMN symbol SET STATISTICS 500;

CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price NUMERIC(15, 4) NOT NULL,
    volume INTEGER,
    side VARCHAR(4) -- 'buy' or 'sell'
) WITH (FILLFACTOR = 100);

CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    cash_balance NUMERIC(15, 2) DEFAULT 0.00 CHECK (cash_balance >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
) WITH (FILLFACTOR = 90);

-- AUDIT: Track portfolio changes
DROP TRIGGER IF EXISTS audit_portfolios ON portfolios;
CREATE TRIGGER audit_portfolios
    AFTER INSERT OR UPDATE OR DELETE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE INDEX IF NOT EXISTS idx_portfolios_user_created ON portfolios(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_portfolios_user_name ON portfolios(user_id, name);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    strike NUMERIC(12, 2),
    expiry DATE,
    option_type VARCHAR(4) CHECK (option_type IS NULL OR option_type IN ('call', 'put')),
    quantity INTEGER NOT NULL CHECK (quantity != 0),
    entry_price NUMERIC(12, 4) NOT NULL,
    entry_date TIMESTAMPTZ DEFAULT NOW(),
    current_price NUMERIC(12, 4),
    exit_price NUMERIC(12, 4),
    exit_date TIMESTAMPTZ,
    realized_pnl NUMERIC(15, 2),
    status VARCHAR(10) DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    CONSTRAINT exit_price_requires_exit_date CHECK (
        (exit_price IS NULL AND exit_date IS NULL) OR
        (exit_price IS NOT NULL AND exit_date IS NOT NULL)
    )
) WITH (
    FILLFACTOR = 90,
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status ON positions(portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_positions_expiry_status ON positions(expiry, status);

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
) WITH (
    FILLFACTOR = 90,
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

-- AUDIT: Track order changes
DROP TRIGGER IF EXISTS audit_orders ON orders;
CREATE TRIGGER audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE INDEX IF NOT EXISTS idx_orders_user_created ON orders(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_created ON orders(portfolio_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_broker_lookup ON orders(broker, broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status, created_at DESC);

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

-- AUDIT: Track model changes
DROP TRIGGER IF EXISTS audit_ml_models ON ml_models;
CREATE TRIGGER audit_ml_models
    AFTER INSERT OR UPDATE OR DELETE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE INDEX IF NOT EXISTS idx_ml_models_production ON ml_models(name, is_production);
CREATE INDEX IF NOT EXISTS idx_ml_models_version ON ml_models(name, version DESC);
CREATE INDEX IF NOT EXISTS idx_ml_models_created_by ON ml_models(created_by);

DROP TABLE IF EXISTS model_predictions CASCADE;
CREATE TABLE model_predictions (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    input_features JSONB NOT NULL,
    predicted_price NUMERIC(12, 4) NOT NULL,
    actual_price NUMERIC(12, 4),
    prediction_error NUMERIC(12, 4),
    actual_value NUMERIC,
    PRIMARY KEY (id, timestamp)
) WITH (FILLFACTOR = 100);

CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_time ON model_predictions(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_time ON model_predictions(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_pending ON model_predictions(timestamp DESC) WHERE actual_price IS NULL;

CREATE TABLE IF NOT EXISTS rate_limits (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(100) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, endpoint, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup ON rate_limits(user_id, endpoint, window_start);

CREATE TABLE IF NOT EXISTS request_logs (
    created_at TIMESTAMPTZ NOT NULL, 
    status_code SMALLINT, 
    path TEXT, 
    method VARCHAR(10), 
    duration_ms DOUBLE PRECISION
) WITH (FILLFACTOR = 100);

CREATE TABLE IF NOT EXISTS model_drift_baselines (
    model_id UUID PRIMARY KEY REFERENCES ml_models(id) ON DELETE CASCADE,
    baseline_accuracy DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_drift_baseline()
RETURNS TRIGGER AS $$
DECLARE
    v_accuracy DOUBLE PRECISION;
BEGIN
    -- Calculate rolling accuracy for the model (predictions from the last 24h)
    SELECT AVG(CASE WHEN ABS(predicted_price - actual_price) / NULLIF(actual_price, 0) < 0.05 THEN 1 ELSE 0 END)
    INTO v_accuracy
    FROM model_predictions
    WHERE model_id = NEW.model_id
      AND timestamp >= NOW() - INTERVAL '24 hours'
      AND actual_price IS NOT NULL;

    -- Update baseline if accuracy is statistically significant
    IF v_accuracy IS NOT NULL AND v_accuracy > 0.90 THEN
        INSERT INTO model_drift_baselines (model_id, baseline_accuracy, updated_at)
        VALUES (NEW.model_id, v_accuracy, NOW())
        ON CONFLICT (model_id) DO UPDATE 
        SET baseline_accuracy = EXCLUDED.baseline_accuracy, updated_at = NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_drift_baseline ON model_predictions;
CREATE TRIGGER trigger_update_drift_baseline
    AFTER UPDATE OF actual_price ON model_predictions
    FOR EACH ROW
    WHEN (NEW.actual_price IS NOT NULL)
    EXECUTE FUNCTION update_drift_baseline();

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
