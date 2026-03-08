# PostgreSQL God-Mode Optimization Guide (BS-OPT Revamp)

This guide documents the advanced PostgreSQL 16 + TimescaleDB optimizations implemented in the BS-OPT platform and explains why many teams skip them.

## 1. Research: Why Others Skip These?

| Optimization | Why others skip it? | The Reality |
| :--- | :--- | :--- |
| **Huge Pages** | Requires host-level configuration and root access. `try` is safe, but `on` can prevent DB startup if misconfigured. | Cold-start performance and memory management are significantly better with Huge Pages enabled. |
| **`pg_prewarm`** | "Postgres warms up automatically as users query." | In high-frequency trading (HFT), waiting for the cache to warm up means failing P99 latency targets for the first 10-20 minutes after a deploy. |
| **`auto_explain`** | "It generates too many logs" or "We use a debugger for queries." | Query plans change based on data volume. `auto_explain` captures plans for *actual* slow production queries that you can't always replicate in dev. |
| **Kernel Tuning** | "Docker handles it" or "It's too risky to touch sysctls." | Default kernel limits for open files (`nofile`) and shared memory are often too low for high-concurrency databases. |
| **Multivariate Stats** | "Postgres' standard stats are enough." | Standard stats assume columns are independent. In trading, `symbol`, `strike`, and `expiry` are highly correlated. Without `CREATE STATISTICS`, the planner significantly underestimates query costs. |
| **Async Commits** | "We can't afford to lose any data." | For market ticks and ephemeral predictions, 100% durability is less important than 10x throughput. We use `synchronous_commit = off` selectively for speed. |

## 2. Implemented "God Mode" Optimizations

### Extensions & Monitoring
- **`pg_prewarm`**: Automatically pre-loads hot tables into memory.
- **`auto_explain`**: Logs execution plans for queries slower than 500ms.
- **`pg_stat_statements`**: Tracks query performance across all sessions.

### Schema & Indexing
- **Multivariate Statistics**: `CREATE STATISTICS` added for `options_prices` (symbol, strike, expiry) and `positions`.
- **GIN Fast Update**: Enabled for JSONB indexes to handle high-frequency hyperparameter and metadata writes.
- **Partial Indexing**: Active positions and open orders have dedicated partial indexes to keep lookups O(1).

### Kernel & OS (via Docker)
- **`sysctls`**: Increased `somaxconn` and `tcp_max_syn_backlog` for high-concurrency networking.
- **`ulimits`**: Locked memory (`memlock: -1`) to prevent Postgres shared memory from being swapped out.
- **`shm_size`**: Increased to 1GB to support large `shared_buffers`.

## 3. How to Validate

1.  **Check Extensions**:
    ```sql
    SELECT * FROM pg_extension;
    ```
2.  **Verify Pre-warming**:
    ```sql
    CALL god_mode_prewarm();
    ```
3.  **Check Statistics**:
    ```sql
    SELECT stxname, stxkind, stxndistinct FROM pg_statistic_ext;
    ```
4.  **Run Benchmarks**:
    ```bash
    python scripts/benchmark_db.py
    ```
