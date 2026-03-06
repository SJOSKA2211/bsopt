-- ============================================================================
-- Black-Scholes Option Pricing Platform - Extensions
-- ============================================================================

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE bsopt_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bsopt_test')\gexec

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
-- Common Enums
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'option_type') THEN
        CREATE TYPE option_type AS ENUM ('call', 'put');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_side') THEN
        CREATE TYPE order_side AS ENUM ('buy', 'sell');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'position_status') THEN
        CREATE TYPE position_status AS ENUM ('open', 'closed', 'liquidated');
    END IF;
END $$;
