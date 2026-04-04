import asyncio

import structlog
from sqlalchemy import text

logger = structlog.get_logger()


async def check_database():
    print("Checking Database [PG16]...", end=" ", flush=True)
    try:
        try:
            from src.database import db_manager
        except ImportError as e:
            print(f"❌ [MISSING DEPENDENCY: {e}]")
            return

        db_manager.initialize()
        engine = db_manager.engine

        with engine.connect() as conn:
            res = conn.execute(
                text("SELECT count(*) FROM pg_views WHERE viewname = 'db_health_overview'")
            ).scalar()
            if res > 0:
                print(" [PG16 HIGH-PERFORMANCE ACTIVE]")
                stats = conn.execute(text("SELECT * FROM db_health_overview")).mappings().first()
                if stats:
                    print(f"   Postgres Version: {stats['pg_version'].split(',')[0]}")
                    print(f"   TimescaleDB Version: {stats['timescale_version']}")
                    print(f"   DB Size: {stats['db_size']}")
                    print(f"   Connections: {stats['total_backends']} total ({stats['active_backends']} active, {stats['idle_backends']} idle, {stats['waiting_backends']} waiting)")

                # Report Timescale specific health
                ts_res = conn.execute(
                    text("SELECT count(*) FROM pg_views WHERE viewname = 'timescale_health_overview'")
                ).scalar()
                if ts_res > 0:
                    ts_stats = conn.execute(text("SELECT * FROM timescale_health_overview")).mappings().all()
                    print("   Hypertables Optimized:")
                    for h in ts_stats:
                        status = "COMPRESSED" if h['compression_enabled'] else "READY"
                        print(f"     - {h['hypertable_name']}: {h['num_chunks']} chunks, {status} ({h['compression_ratio_pct']}% saved)")
            else:
                print("⚠️ [SCHEMA OK, REVAMP DIAGNOSTICS MISSING]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")


async def check_pgbouncer():
    print("Checking PgBouncer Pool Engine...", end=" ", flush=True)
    import os
    try:
        import psycopg
    except ImportError:
        print(" ❌ [FAILED: psycopg (v3) not installed]")
        return

    from src.shared.config import settings

    host = os.environ.get("PGBOUNCER_HOST", settings.PGBOUNCER_HOST)
    port = int(os.environ.get("PGBOUNCER_PORT", settings.PGBOUNCER_PORT))
    sslmode = os.environ.get("PGBOUNCER_SSLMODE", "require")

    admin_url = f"postgresql://{settings.PGBOUNCER_ADMIN_USER}:{settings.PGBOUNCER_ADMIN_PASSWORD}@{host}:{port}/pgbouncer?sslmode={sslmode}"

    try:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute("SHOW POOLS")
                pools = cur.fetchall()
                
                if not pools:
                    print(" [ALIVE: No pools]")
                    return

                # Summarize across all pools
                cl_active = sum(p['cl_active'] for p in pools)
                cl_waiting = sum(p['cl_waiting'] for p in pools)
                sv_active = sum(p['sv_active'] for p in pools)
                sv_idle = sum(p['sv_idle'] for p in pools)

                status = "HEALTHY"
                if cl_waiting > 0:
                    status = "⚠️ UNHEALTHY (CLIENTS WAITING)"
                
                print(f" [{status}]")
                print(f"   Clients: {cl_active} active, {cl_waiting} waiting")
                print(f"   Servers: {sv_active} active, {sv_idle} idle")
                
                cur.execute("SHOW STATS")
                stats = cur.fetchall()
                if stats:
                    total_xact = sum(s['total_xact_count'] for s in stats)
                    total_wait = sum(s['total_wait_time'] for s in stats)
                    print(f"   Throughput: {total_xact} xacts total, {total_wait}us total wait")

    except Exception as e:
        print(f" ❌ [FAILED: {e}]")


async def check_redis():
    print("Checking Redis Cluster...", end=" ", flush=True)
    try:
        import socket

        try:
            from src.shared.utils.cache import get_redis
        except ImportError as e:
            print(f"❌ [MISSING DEPENDENCY: {e}]")
            return

        try:
            socket.gethostbyname("redis")
            # In docker
            redis = get_redis()  # uses environment variable
        except Exception as e:
            # Outside docker, try localhost or REDIS_URL
            import os
            try:
                import redis.asyncio as aioredis
            except ImportError:
                print(" ❌ [FAILED: redis-py not installed]")
                return
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            logger.warning("redis_connection_fallback", error=str(e), using_url=redis_url)
            redis = aioredis.from_url(redis_url, decode_responses=True)

        if redis:
            await redis.set("sentinel_ping", "pong")
            val = await redis.get("sentinel_ping")
            if val == "pong":
                print(" [ALIVE]")
            else:
                print("⚠️ [UNEXPECTED RESPONSE]")
        else:
            print("❌ [DISCONNECTED]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")


async def check_shm():
    print("Checking Shared Memory Mesh...", end=" ", flush=True)
    try:
        import os
        from multiprocessing import shared_memory

        from src.shared.shm_init import SHM_CONFIGS

        missing = []
        # On Linux, shared memory segments appear in /dev/shm
        # Python's SharedMemory(name='x') creates '/dev/shm/x'
        dev_shm = "/dev/shm"
        shm_files = os.listdir(dev_shm) if os.path.exists(dev_shm) else []
        
        for config in SHM_CONFIGS:
            name = config["name"]
            # Check if it exists via multiprocessing API
            try:
                shm = shared_memory.SharedMemory(name=f"/{name}")
                shm.close()
            except FileNotFoundError:
                # Fallback: check /dev/shm directly
                if not any(f == name or f == f"/{name}" for f in shm_files) and not any(f == name.lstrip("/") for f in shm_files):
                    missing.append(name)

        if not missing:
            print(" [PRESSURIZED]")
        else:
            print(f"⚠️ [MISSING/CORRUPT: {', '.join(missing)}]")
            if shm_files:
                # Filter out semaphores and other unrelated files
                actual_shm = [f for f in shm_files if not f.startswith("sem.")]
                if actual_shm:
                    print(f"   (Detected in /dev/shm: {', '.join(actual_shm[:5])})")
    except Exception as e:
        print(f"❌ [ERROR: {e}]")


async def main():
    print("\n" + "=" * 50)
    print("   BS-OPT HIGH-PERFORMANCE SYSTEM SENTINEL")
    print("=" * 50)
    await check_database()
    await check_pgbouncer()
    await check_redis()
    await check_shm()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
