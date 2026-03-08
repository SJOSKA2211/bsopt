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
        engine = get_engine()
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
        redis = get_redis()
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
        try:
            shm_risk = shared_memory.SharedMemory(name=SHM_RISK_NAME)
            shm_risk.close()
            shm_order = shared_memory.SharedMemory(name=SHM_ORDER_NAME)
            shm_order.close()
            print("✅ [PRESSURIZED]")
        except FileNotFoundError:
            print("⚠️ [SHM BUFFERS NOT INITIALIZED]")
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
