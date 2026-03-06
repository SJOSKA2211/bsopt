-- ============================================================================
-- Black-Scholes Option Pricing Platform - Market Data Mesh
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_data_mesh (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    open NUMERIC(12, 4) CHECK (open >= 0),
    high NUMERIC(12, 4) CHECK (high >= 0),
    low NUMERIC(12, 4) CHECK (low >= 0),
    close NUMERIC(12, 4) NOT NULL CHECK (close >= 0),
    volume BIGINT CHECK (volume >= 0),
    source_type TEXT NOT NULL,
    metadata JSONB
) WITH (FILLFACTOR = 100);

SELECT create_hypertable('market_data_mesh', 'time', if_not_exists => TRUE);
SELECT add_dimension('market_data_mesh', 'symbol', number_partitions => 4, if_not_exists => TRUE);

-- Compression Policy
ALTER TABLE market_data_mesh SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('market_data_mesh', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention Policy
SELECT add_retention_policy('market_data_mesh', INTERVAL '1 year', if_not_exists => TRUE);
