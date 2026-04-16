import asyncio
import os
import time

import structlog
from sqlalchemy import text

logger = structlog.get_logger()


async def check_network():
    print("Checking Secure Network Layer...", end=" ", flush=True)
    import ssl
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        # Check Nginx (HTTPS Redirect)
        with urllib.request.urlopen("https://localhost:8443/health", context=ctx, timeout=2) as r:
            nginx_status = "HEALTHY" if r.getcode() == 200 else f"UNEXPECTED ({r.getcode()})"
        
        # Check Envoy Admin
        with urllib.request.urlopen("http://localhost:9901/ready", timeout=2) as r:
            envoy_status = "LIVE" if "LIVE" in r.read().decode() else "NOT READY"
            
        print(f" [Nginx: {nginx_status} | Envoy: {envoy_status}]")
    except Exception as e:
        print(f" [FAILED: {e}]")


async def check_database():
    print("Checking Database [PG16]...", end=" ", flush=True)
    start = time.time()
    try:
        try:
            from src.database import db_manager
        except ImportError as e:
            print(f" [MISSING DEPENDENCY: {e}]")
            return

        db_manager.initialize()
        engine = db_manager.engine

        with engine.connect() as conn:
            latency = (time.time() - start) * 1000
            res = conn.execute(
                text("SELECT count(*) FROM pg_views WHERE viewname = 'db_health_overview'")
            ).scalar()
            if res > 0:
                print(f" [ALIVE | RTT: {latency:.1f}ms]")
                stats = conn.execute(text("SELECT * FROM db_health_overview")).mappings().first()
                # Performance stats might not be implemented yet in revamp_db_views
                print(f"   Connections: {stats['active_backends']} active")
                print(f"   Version: {stats['pg_version']}")
            else:
                # Check for the other view from engine_revamp_god_mode
                res = conn.execute(
                    text("SELECT count(*) FROM pg_views WHERE viewname = 'db_engine_health'")
                ).scalar()
                if res > 0:
                    print(f" [ALIVE | RTT: {latency:.1f}ms]")
                    stats = conn.execute(text("SELECT * FROM db_engine_health")).mappings().first()
                    print(f"   Size: {stats['total_size']} | Cache Hit: {float(stats['cache_hit_ratio'])*100:.2f}%")
                else:
                    print("️ [ALIVE | REVAMP DIAGNOSTICS MISSING]")
    except Exception as e:
        print(f" [FAILED: {e}]")


async def check_pgbouncer():
    print("Checking PgBouncer Pool Engine...", end=" ", flush=True)
    import psycopg

    from src.shared.config import settings
    
    host = os.environ.get('PGBOUNCER_HOST', settings.PGBOUNCER_HOST)
    port = int(os.environ.get('PGBOUNCER_PORT', settings.PGBOUNCER_PORT))
    
    conn_str = f"host={host} port={port} user={settings.PGBOUNCER_ADMIN_USER} password={settings.PGBOUNCER_ADMIN_PASSWORD} dbname=pgbouncer"
    
    try:
        with psycopg.connect(conn_str, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW POOLS")
                pools = cur.fetchall()
                
                total_active = 0
                total_waiting = 0
                # SHOW POOLS columns: database, user, cl_active, cl_waiting, sv_active, sv_idle, ...
                # psycopg fetchall returns list of tuples. We need to find the right indices.
                # Usually: database(0), user(1), cl_active(2), cl_waiting(3)
                for pool in pools:
                    total_active += pool[2]
                    total_waiting += pool[3]
                
                if total_waiting > 0:
                    print(f"️ [CONGESTED: {total_active} active, {total_waiting} waiting]")
                else:
                    print(f" [HEALTHY: {total_active} active connections] [HEALTHY]")
    except Exception as e:
        print(f" [FAILED: {e}]")


async def check_redis():
    print("Checking Redis Cluster...", end=" ", flush=True)
    start = time.time()
    try:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            print("  [FAILED: redis-py not installed]")
            return
            
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6380/0")
        redis = aioredis.from_url(redis_url, decode_responses=True)

        await redis.set("sentinel_ping", "pong")
        val = await redis.get("sentinel_ping")
        latency = (time.time() - start) * 1000
        if val == "pong":
            print(f" [ALIVE | RTT: {latency:.1f}ms]")
        else:
            print("️ [UNEXPECTED RESPONSE]")
    except Exception as e:
        print(f" [FAILED: {e}]")


async def check_shm():
    print("Checking Shared Memory Mesh...", end=" ", flush=True)
    try:
        from src.shared.shm_init import SHM_CONFIGS

        missing = []
        dev_shm = "/dev/shm"
        shm_files = os.listdir(dev_shm) if os.path.exists(dev_shm) else []
        
        # Calculate pressure
        try:
            import shutil
            total, used, free = shutil.disk_usage(dev_shm)
            pressure = (used / total) * 100
        except:
            pressure = 0

        for config in SHM_CONFIGS:
            name = config["name"]
            try:
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=f"/{name}")
                shm.close()
            except:
                if name not in shm_files:
                    missing.append(name)

        if not missing:
            print(f" [PRESSURIZED | Usage: {pressure:.1f}%]")
        else:
            print(f"️ [MISSING/CORRUPT: {', '.join(missing)}]")
    except Exception as e:
        print(f" [ERROR: {e}]")


async def main():
    print("\n" + "=" * 50)
    print("   BS-OPT HIGH-PERFORMANCE SYSTEM SENTINEL")
    print("=" * 50)
    await check_network()
    await check_database()
    await check_pgbouncer()
    await check_redis()
    await check_shm()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
