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
            # Check for our revamped diagnostic view
            res = conn.execute(
                text("SELECT count(*) FROM pg_views WHERE viewname = 'db_health_overview'")
            ).scalar()
            if res > 0:
                print(" [PG16 HIGH-PERFORMANCE ACTIVE]")
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
            with conn.cursor() as cur:
                cur.execute("SHOW POOLS")
                pools = cur.fetchall()
                if pools:
                    print(f" [HEALTHY: {len(pools)} pools active]")
                else:
                    print(" [ALIVE: No pools]")
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
        from multiprocessing import shared_memory
        import os
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
