import asyncio

import structlog
from sqlalchemy import text

logger = structlog.get_logger()

async def check_database():
    print("Checking Database [PG16]...", end=" ", flush=True)
    try:
        from src.database import db_manager

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
    from sqlalchemy import create_engine, text
    from src.shared.config import settings
    
    # Allow host/port/sslmode overrides for local testing outside docker
    host = os.environ.get("PGBOUNCER_HOST", settings.PGBOUNCER_HOST)
    port = os.environ.get("PGBOUNCER_PORT", settings.PGBOUNCER_PORT)
    sslmode = os.environ.get("PGBOUNCER_SSLMODE", "verify-full")
    
    # Use 'postgresql+psycopg' to force v3 driver
    admin_url = f"postgresql+psycopg://{settings.PGBOUNCER_ADMIN_USER}:{settings.PGBOUNCER_ADMIN_PASSWORD}@{host}:{port}/pgbouncer?sslmode={sslmode}"
    
    try:
        engine = create_engine(admin_url)
        with engine.connect() as conn:
            pools = conn.execute(text("SHOW POOLS")).fetchall()
            
            total_active = 0
            total_waiting = 0
            for pool in pools:
                total_active += pool.cl_active
                total_waiting += pool.cl_waiting
            
            if total_waiting > 0:
                print(f"⚠️ [CONGESTED: {total_active} active, {total_waiting} waiting]")
            else:
                print(f" [HEALTHY: {total_active} active connections]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")

async def check_redis():
    print("Checking Redis Cluster...", end=" ", flush=True)
    try:
        import socket

        from src.shared.utils.cache import get_redis

        try:
            socket.gethostbyname("redis")
            # In docker
            redis = get_redis()  # uses environment variable
        except Exception as e:
            # Outside docker, try localhost
            logger.warning("redis_connection_fallback", error=str(e))
            import redis.asyncio as aioredis

            redis = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)

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

        from src.shared.shm_init import SHM_CONFIGS

        missing = []
        for config in SHM_CONFIGS:
            name = config["name"]
            try:
                shm = shared_memory.SharedMemory(name=name)
                if shm.size != config["size"]:
                    missing.append(f"{name} (size mismatch)")
                shm.close()
            except FileNotFoundError:
                missing.append(name)

        if not missing:
            print(" [PRESSURIZED]")
        else:
            print(f"⚠️ [MISSING/CORRUPT: {', '.join(missing)}]")
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
