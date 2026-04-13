# Design: PgBouncer Health Revamp (Approach 1)

## 1. Problem Statement
The current system lacks deep visibility into PgBouncer's internal state (pooling, connection counts, wait times). While basic process health is checked, we need a "Revamp" that provides granular metrics to ensure the "Engine" is healthy and high-performing.

## 2. Approach: Integrated Sentinel Diagnostics
We will enhance the existing `scripts/system_sentinel.py` to connect directly to the PgBouncer admin console. This allows us to report real-time pooling statistics alongside existing database and shared memory health checks.

## 3. Architecture Changes

### 3.1. System Sentinel Enhancement
- **PgBouncer Admin Connection**: Add logic to connect to the `pgbouncer` pseudo-database using the SQLAlchemy engine or a raw `psycopg` connection.
- **Pooling Metrics**: Execute `SHOW POOLS` and `SHOW STATS` commands.
- **Health Reporting**:
    - **Active Clients**: `cl_active`
    - **Waiting Clients**: `cl_waiting` (Critical indicator of pool exhaustion)
    - **Server Connections**: `sv_active`, `sv_idle`
    - **Transaction Stats**: Total requests, total wait time.

### 3.2. Database Revamp (Diagnostics View)
- Ensure the `db_health_overview` view exists in the main database to provide a high-level summary of TimescaleDB and general PG health.

## 4. Implementation Details

### PgBouncer Admin Connector
```python
async def check_pgbouncer():
    # Connect to pgbouncer db on port 6432
    # Stats to track: cl_active, cl_waiting, sv_active, sv_idle
    pass
```

## 5. Success Criteria
- Sentinel reports PgBouncer connection stats.
- Sentinel flags "Unhealthy" if `cl_waiting` > 0 for a sustained period.
- Revamp diagnostic view is active and reporting in Postgres.
