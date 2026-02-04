-- 🚀 Optimized TimescaleDB Hypertables for Market Data
-- Converts standard tables to hypertables for time-series optimization.

-- 1. Options Prices Hypertable
-- Partitioning by 'time' (required) and 'symbol' (recommended for partitioning across symbols)
SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);

-- 2. Market Ticks Hypertable
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);

-- 3. Compression Policies
-- Options data is highly repetitive, so compression is extremely effective.
ALTER TABLE options_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);
-- -- SELECT add_compression_policy('options_prices', INTERVAL '1 day');

ALTER TABLE market_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);
-- -- SELECT add_compression_policy('market_ticks', INTERVAL '1 day');

-- 4. Retention Policies
-- Keep granular price data for 1 year, keep market ticks for 6 months
-- -- SELECT add_retention_policy('options_prices', INTERVAL '1 year');
-- -- SELECT add_retention_policy('market_ticks', INTERVAL '6 months');
