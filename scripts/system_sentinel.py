import asyncio
import time
import os
import struct
import numpy as np
from sqlalchemy import text
from src.database import get_engine
from src.utils.cache import get_redis
from src.shared.shm_mesh import RiskStateBuffer, SHM_ORDER_NAME, SHM_RISK_NAME
import structlog

logger = structlog.get_logger()

async def check_database():
    print("Checking Database [PG16]...", end=" ", flush=True)
    try:
        from src.config import settings
        # Dynamic host selection for docker environment
        import socket
        try:
            socket.gethostbyname("postgres")
            host = "postgres"
        except:
            host = "localhost"
            
        from sqlalchemy import create_engine
        # Re-derive URL for sentinel check
        url = settings.DATABASE_URL.replace("postgres", host) if "postgres" in settings.DATABASE_URL else settings.DATABASE_URL
        engine = create_engine(url)
        
        with engine.connect() as conn:
            # Check for PG16 diagnostic view
            res = conn.execute(text("SELECT count(*) FROM pg_views WHERE viewname = 'system_wait_bottlenecks'")).scalar()
            if res > 0:
                print("✅ [PG16 GOD-MODE ACTIVE]")
            else:
                print("⚠️ [SCHEMA OK, DIAGNOSTICS MISSING]")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")

async def check_redis():
    print("Checking Redis Cluster...", end=" ", flush=True)
    try:
        from src.utils.cache import get_redis
        import socket
        try:
            socket.gethostbyname("redis")
            # In docker
            redis = get_redis() # uses environment variable
        except:
            # Outside docker, try localhost
            import redis.asyncio as aioredis
            redis = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
            
        if redis:
            await redis.set("sentinel_ping", "pong")
            val = await redis.get("sentinel_ping")
            if val == "pong":
                print("✅ [ALIVE]")
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
            print("✅ [PRESSURIZED]")
        else:
            print(f"⚠️ [MISSING/CORRUPT: {', '.join(missing)}]")
    except Exception as e:
        print(f"❌ [ERROR: {e}]")

async def main():
    print("\n" + "="*50)
    print("   BS-OPT GOD-MODE SYSTEM SENTINEL")
    print("="*50)
    await check_database()
    await check_redis()
    await check_shm()
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
