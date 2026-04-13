# PgBouncer Health Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate PgBouncer internal pooling metrics into the system sentinel and revamp the database diagnostics view.

**Architecture:** Approach 1 (Integrated Sentinel) connects to the 'pgbouncer' admin database to report client/server connection states. A new SQL view provides deep database health insights.

**Tech Stack:** Python (SQLAlchemy, structlog), PostgreSQL (PgBouncer, TimescaleDB).

---

### Task 1: PgBouncer Admin Credentials in Settings

**Files:**
- Modify: `src/shared/config.py`

- [ ] **Step 1: Add PgBouncer admin settings**

Add the following fields to the `Settings` class:
```python
    PGBOUNCER_ADMIN_USER: str = Field(default="admin", validation_alias="PGBOUNCER_ADMIN_USER")
    PGBOUNCER_ADMIN_PASSWORD: str = Field(default="password", validation_alias="PGBOUNCER_ADMIN_PASSWORD")
    PGBOUNCER_HOST: str = Field(default="pgbouncer", validation_alias="PGBOUNCER_HOST")
    PGBOUNCER_PORT: int = Field(default=6432, validation_alias="PGBOUNCER_PORT")
```

- [ ] **Step 2: Commit settings changes**

```bash
git add src/shared/config.py
git commit -m "feat: add pgbouncer admin settings to config"
```

### Task 2: Integrated Sentinel Revamp

**Files:**
- Modify: `scripts/system_sentinel.py`

- [ ] **Step 1: Implement `check_pgbouncer` function**

Add this function to `scripts/system_sentinel.py`:
```python
async def check_pgbouncer():
    print("Checking PgBouncer Pool Engine...", end=" ", flush=True)
    from sqlalchemy import create_engine, text
    from src.shared.config import settings
    
    # Connect to the special 'pgbouncer' database
    admin_url = f"postgresql://{settings.PGBOUNCER_ADMIN_USER}:{settings.PGBOUNCER_ADMIN_PASSWORD}@{settings.PGBOUNCER_HOST}:{settings.PGBOUNCER_PORT}/pgbouncer"
    
    try:
        # Use a temporary engine for the admin check
        engine = create_engine(admin_url)
        with engine.connect() as conn:
            # SHOW POOLS returns statistics about each pool
            pools = conn.execute(text("SHOW POOLS")).fetchall()
            
            total_active = 0
            total_waiting = 0
            for pool in pools:
                # pool fields: database, user, cl_active, cl_waiting, sv_active, sv_idle, ...
                total_active += pool.cl_active
                total_waiting += pool.cl_waiting
            
            if total_waiting > 0:
                print(f" [CONGESTED: {total_active} active, {total_waiting} waiting]")
            else:
                print(f" [HEALTHY: {total_active} active connections]")
    except Exception as e:
        print(f" [FAILED: {e}]")
```

- [ ] **Step 2: Update `main` to include PgBouncer check**

```python
async def main():
    print("\n" + "=" * 50)
    print("   BS-OPT HIGH-PERFORMANCE SYSTEM SENTINEL")
    print("=" * 50)
    await check_database()
    await check_pgbouncer()  # New check
    await check_redis()
    await check_shm()
    print("=" * 50 + "\n")
```

- [ ] **Step 3: Test the sentinel**

Run: `PYTHONPATH=. python3 scripts/system_sentinel.py`
Expected: Output includes "Checking PgBouncer Pool Engine... [HEALTHY: ...]"

- [ ] **Step 4: Commit sentinel changes**

```bash
git add scripts/system_sentinel.py
git commit -m "feat: integrate pgbouncer pool metrics into system sentinel"
```

### Task 3: Database Diagnostics View Revamp

**Files:**
- Create: `scripts/revamp_db_views.py`

- [ ] **Step 1: Create script to initialize the diagnostics view**

```python
import structlog
from sqlalchemy import text
from src.database import db_manager

logger = structlog.get_logger()

def revamp_diagnostics():
    db_manager.initialize()
    engine = db_manager.engine
    
    view_sql = """
    CREATE OR REPLACE VIEW db_health_overview AS
    SELECT
        now() as check_time,
        (SELECT count(*) FROM pg_stat_activity) as active_backends,
        (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL) as waiting_backends,
        (SELECT version()) as pg_version,
        (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') as timescale_version;
    """
    
    with engine.connect() as conn:
        conn.execute(text(view_sql))
        conn.commit()
        print(" Database diagnostics view 'db_health_overview' REVAMPED.")

if __name__ == "__main__":
    revamp_diagnostics()
```

- [ ] **Step 2: Run the revamp script**

Run: `PYTHONPATH=. python3 scripts/revamp_db_views.py`
Expected: " Database diagnostics view 'db_health_overview' REVAMPED."

- [ ] **Step 3: Verify with sentinel**

Run: `PYTHONPATH=. python3 scripts/system_sentinel.py`
Expected: "Checking Database [PG16]... [PG16 HIGH-PERFORMANCE ACTIVE]"

- [ ] **Step 4: Commit database revamp**

```bash
git add scripts/revamp_db_views.py
git commit -m "feat: implement database diagnostics view revamp"
```
