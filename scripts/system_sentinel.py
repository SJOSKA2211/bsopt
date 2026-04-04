import asyncio

import structlog
from sqlalchemy import text

logger = structlog.get_logger()


async def check_network():
    print("Checking Secure Network Layer...", end=" ", flush=True)
    import urllib.request
    import ssl
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
        print(f"❌ [FAILED: {e}]")


async def check_database():
    print("Checking Database [PG16]...", end=" ", flush=True)
    start = time.time()
    try:
        try:
            from src.database import db_manager
        except ImportError as e:
            print(f"❌ [MISSING DEPENDENCY: {e}]")
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
                perf = conn.execute(text("SELECT * FROM db_performance_stats")).mappings().first()
                
                if stats and perf:
                    print(f"   Connections: {stats['total_backends']} total ({stats['active_backends']} active)")
                    print(f"   Cache Hit: {perf['heap_cache_hit_ratio']}% (Heap), {perf['index_cache_hit_ratio']}% (Index)")
                    if perf['blocked_queries'] > 0:
                        print(f"   ⚠️ WARNING: {perf['blocked_queries']} blocked queries detected")
            else:
                print("⚠️ [ALIVE | REVAMP DIAGNOSTICS MISSING]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")


async def check_redis():
    print("Checking Redis Cluster...", end=" ", flush=True)
    start = time.time()
    try:
        import os
        try:
            import redis.asyncio as aioredis
        except ImportError:
            print(" ❌ [FAILED: redis-py not installed]")
            return
            
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6380/0")
        redis = aioredis.from_url(redis_url, decode_responses=True)

        await redis.set("sentinel_ping", "pong")
        val = await redis.get("sentinel_ping")
        latency = (time.time() - start) * 1000
        if val == "pong":
            print(f" [ALIVE | RTT: {latency:.1f}ms]")
        else:
            print("⚠️ [UNEXPECTED RESPONSE]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")


async def check_shm():
    print("Checking Shared Memory Mesh...", end=" ", flush=True)
    try:
        import os
        from multiprocessing import shared_memory
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
                shm = shared_memory.SharedMemory(name=f"/{name}")
                shm.close()
            except:
                if name not in shm_files:
                    missing.append(name)

        if not missing:
            print(f" [PRESSURIZED | Usage: {pressure:.1f}%]")
        else:
            print(f"⚠️ [MISSING/CORRUPT: {', '.join(missing)}]")
    except Exception as e:
        print(f"❌ [ERROR: {e}]")


async def main():
    import os
    print("\n" + "=" * 50)
    print("   BS-OPT HIGH-PERFORMANCE SYSTEM SENTINEL")
    print("=" * 50)
    await check_network()
    await check_database()
    await check_redis()
    await check_shm()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
