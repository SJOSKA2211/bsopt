-- ============================================================================
-- Black-Scholes Option Pricing Platform - Market Data Mesh
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_data_mesh (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT,
    source_type TEXT NOT NULL,
    metadata JSONB
);

SELECT create_hypertable('market_data_mesh', 'time', if_not_exists => TRUE);
