# Plan: Optimized PostgreSQL/TimescaleDB Architecture for BS-Opt v4.0

## Phase 1: Infrastructure & Migration Setup (TDD) [checkpoint: 33c1d84]
- [x] Task: Configure Docker Compose for Optimized PostgreSQL. 3e1358b
    - [ ] Sub-task: Write tests to verify PostgreSQL container starts with target extensions (TimescaleDB, pgvector).
    - [ ] Sub-task: Implement `command` overrides in `docker-compose.yml` for performance tuning.
- [x] Task: Initialize Alembic for Migration Management. 6001d1b
    - [ ] Sub-task: Set up Alembic environment and connection string.
    - [ ] Sub-task: Create initial migration script to enable required extensions.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Migration Setup' (Protocol in workflow.md) 33c1d84

## Phase 2: Market Data Module (TDD) [checkpoint: 40f3e5e]
- [x] Task: Implement `market_ticks` Hypertable. 9ea6c2f
    - [ ] Sub-task: Write failing tests for tick data insertion and retrieval.
    - [ ] Sub-task: Create Alembic migration for `market_ticks` table and convert to hypertable.
- [x] Task: Configure Compression and Continuous Aggregates. a4846bf
    - [ ] Sub-task: Write tests to verify OHLCV aggregation via `market_candles_1m`.
    - [ ] Sub-task: Create Alembic migration for compression policy and continuous aggregate view.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Market Data Module' (Protocol in workflow.md) 40f3e5e

## Phase 3: Options Chain & Pricing Module (TDD) [checkpoint: ee0b8de]
- [x] Task: Implement Options Metadata and Indexing. a9484af
    - [ ] Sub-task: Write failing tests for searching options by underlying/expiry/strike.
    - [ ] Sub-task: Create Alembic migration for `option_contracts` with optimized indices.
- [x] Task: Implement `option_greeks` Hypertable. e90f81d
    - [ ] Sub-task: Write failing tests for high-frequency pricing data ingest.
    - [ ] Sub-task: Create Alembic migration for `option_greeks` hypertable.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Options Chain & Pricing Module' (Protocol in workflow.md) ee0b8de

## Phase 4: AI/ML Model Store (pgvector) (TDD)
- [ ] Task: Implement Vector Embedding Store.
    - [ ] Sub-task: Write failing tests for vector similarity search (L2 distance).
    - [ ] Sub-task: Create Alembic migration for `model_embeddings` with HNSW index.
- [ ] Task: Implement RL Episode Tracking.
    - [ ] Sub-task: Write tests for storing and retrieving RL agent hyperparameters and performance metrics.
    - [ ] Sub-task: Create Alembic migration for `rl_episodes` table.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: AI/ML Model Store (pgvector)' (Protocol in workflow.md)

## Phase 5: Portfolio & Trading Module (RLS) (TDD)
- [ ] Task: Implement Transactional Core.
    - [ ] Sub-task: Write failing tests for portfolio and position management.
    - [ ] Sub-task: Create Alembic migration for `users`, `portfolios`, and `positions` tables.
- [ ] Task: Enforce Row Level Security (RLS).
    - [ ] Sub-task: Write failing tests to verify that User A cannot see User B's portfolio data.
    - [ ] Sub-task: Create Alembic migration to enable RLS and define security policies.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Portfolio & Trading Module (RLS)' (Protocol in workflow.md)
