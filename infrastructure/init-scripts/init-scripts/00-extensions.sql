-- ============================================================================
-- Black-Scholes Option Pricing Platform - Extensions & Core Types
-- ============================================================================

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE bsopt_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bsopt_test')\gexec

-- Enable required extensions (High-Performance Standard)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

--  HIGH-PERFORMANCE: Consolidated Enum Types
DO $$
BEGIN
    -- User Tiers
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_tier') THEN
        CREATE TYPE user_tier AS ENUM ('free', 'pro', 'enterprise');
    END IF;

    -- Trading Sides
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_side') THEN
        CREATE TYPE order_side AS ENUM ('buy', 'sell');
    END IF;

    -- Order Statuses
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_status') THEN
        CREATE TYPE order_status AS ENUM ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected');
    END IF;

    -- Order Types
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_type') THEN
        CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
    END IF;

    -- Position Statuses
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'position_status') THEN
        CREATE TYPE position_status AS ENUM ('open', 'closed', 'liquidated');
    END IF;

    -- Option Types
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'option_type') THEN
        CREATE TYPE option_type AS ENUM ('call', 'put');
    END IF;

    -- ML Algorithms
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ml_algorithm') THEN
        CREATE TYPE ml_algorithm AS ENUM ('xgboost', 'lightgbm', 'neural_network', 'random_forest', 'svm', 'ensemble');
    END IF;
END $$;
