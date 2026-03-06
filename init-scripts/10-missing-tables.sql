-- ============================================================================
-- Black-Scholes Option Pricing Platform - Missing Tables
-- ============================================================================

-- 1. Option Contracts Metadata
CREATE TABLE IF NOT EXISTS option_contracts (
    id TEXT PRIMARY KEY,
    underlying TEXT NOT NULL,
    expiry DATE NOT NULL,
    strike NUMERIC(12, 2) NOT NULL,
    option_type option_type NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_option_contracts_underlying_expiry_strike 
ON option_contracts (underlying, expiry, strike);

-- 2. Model Embeddings (pgvector)
CREATE TABLE IF NOT EXISTS model_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Optimization: Fast lookup for model-specific embeddings
CREATE INDEX IF NOT EXISTS idx_model_embeddings_lookup 
ON model_embeddings (model_id, version);

-- HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_model_embeddings_vector 
ON model_embeddings USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 64);

-- 3. RL Episodes
CREATE TABLE IF NOT EXISTS rl_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    episode_reward DOUBLE PRECISION NOT NULL,
    steps INTEGER NOT NULL,
    hyperparameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_episodes_agent ON rl_episodes(agent_id);

-- Permissions for app_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE option_contracts TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE model_embeddings TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE rl_episodes TO app_user;
