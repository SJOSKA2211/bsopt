# PostgreSQL Performance Tuning & Monitoring

This directory contains scripts and guides for monitoring and tuning the PostgreSQL database.

## `work_mem` Tuning

The `work_mem` setting in PostgreSQL specifies the amount of memory to be used by internal sort operations and hash tables before writing to temporary disk files. The default value is often conservative.

### `work_mem_monitoring.sql`

This script uses the `pg_stat_statements` extension to identify queries that are spilling to disk (i.e., using temporary files). This is a strong indicator that `work_mem` is not high enough for your workload.

**How to use:**

1.  Ensure `pg_stat_statements` is enabled in your `postgresql.conf` and that the extension is created in your database.
2.  Connect to your database using `psql` or another client.
3.  Run the query in `work_mem_monitoring.sql`.
4.  Analyze the output. If you see queries with a high `temp_blks_written` value, it means they are using a significant amount of temporary disk space.
5.  Consider increasing `work_mem` in `postgresql.conf` (e.g., from `16MB` to `32MB` or `64MB`) and monitor the impact. Be aware that this memory can be used by each sorting operation, so don't set it too high.

## Table Bloat and `pg_repack`

PostgreSQL's MVCC architecture can lead to table and index bloat over time, especially in high-transaction environments. While `VACUUM` reclaims space for reuse, it doesn't return it to the operating system. `VACUUM FULL` does, but it takes an exclusive lock on the table, blocking reads and writes.

### `pg_repack`

The `pg_repack` extension is the recommended way to remove bloat from tables and indexes without holding an exclusive lock. It essentially creates a new, bloat-free copy of the table and then swaps it with the old one.

**How to use:**

1.  Ensure the `pg_repack` extension is created in your database (it has been added to `init-scripts/00-extensions.sql`).
2.  Install the `pg_repack` client tool on a machine that can connect to your database.
3.  Run `pg_repack` from the command line. For example, to repack the `orders` table in the `bsopt` database:

    ```bash
    pg_repack -d bsopt -t orders
    ```

**Recommendation:**

*   Periodically run `pg_repack` on heavily updated tables like `orders` and `positions`. A weekly or monthly schedule is a good starting point.
*   You can also use queries to identify bloated tables and indexes and target them for repacking.
