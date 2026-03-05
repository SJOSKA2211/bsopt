-- ============================================================================
-- Black-Scholes Option Pricing Platform - Extensions
-- ============================================================================

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE bsopt_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bsopt_test')\gexec

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto CASCADE;
CREATE EXTENSION IF NOT EXISTS btree_gist CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
