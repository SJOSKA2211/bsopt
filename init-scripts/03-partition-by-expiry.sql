-- 🚀 DATABASE PERFORMANCE TUNING
-- Optimize options_prices hypertable by adding a space dimension on 'expiry'
-- This improves query performance for specific option chains and enables 
-- better data locality for maturity-based queries.

-- Note: The column must be part of the primary key, which it already is in our schema.
SELECT add_dimension('options_prices', 'expiry', number_partitions => 4, if_not_exists => TRUE);

-- Create a specialized index for maturity-based lookups if not already present
-- This complements the hypertable partitioning
CREATE INDEX IF NOT EXISTS idx_options_prices_expiry_only ON options_prices (expiry DESC);

-- Analyze the table to update statistics
ANALYZE options_prices;
