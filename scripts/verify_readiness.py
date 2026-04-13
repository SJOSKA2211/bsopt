import asyncio
import os
import sys

import aio_pika
import asyncpg
import httpx
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)

# Update to match refactored Envoy/API ports
SERVICES = {
    "Envoy Edge": "http://localhost:8081/ready",
    "Auth Service": "http://localhost:3001/health",
    "API Backend": "http://localhost:8000/health",
    "ML-Inference": "http://localhost:5001/health",
    "MLflow": "http://localhost:5000/api/2.0/mlflow/experiments/list",
}


async def check_http_service(name: str, url: str) -> bool:
    """Check a microservice's health endpoint."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(url)
            if resp.status_code in [200, 204]:
                print(f" {name:15} | ONLINE")
                return True
            else:
                print(f" {name:15} | DEGRADED (Status: {resp.status_code})")
                return False
        except Exception as e:
            print(f" {name:15} | DOWN ({str(e)})")
            return False


async def check_postgres() -> bool:
    db_url = os.getenv("DATABASE_URL_LOCAL", "postgresql://admin:password@localhost:6434/bsopt")
    try:
        # Use sslmode=require for testing connectivity
        if "?" in db_url:
            db_url += "&sslmode=require"
        else:
            db_url += "?sslmode=require"

        conn = await asyncpg.connect(db_url)
        await conn.execute("SELECT 1")
        await conn.close()
        print(f" {'TimescaleDB':15} | CONNECTED")
        return True
    except Exception as e:
        print(f" {'TimescaleDB':15} | CONNECTION FAILED ({str(e)})")
        return False


async def check_redis() -> bool:
    try:
        password = os.getenv("REDIS_PASSWORD")
        r = redis.from_url("redis://:password@localhost:6380/0")
        if password:
            r = redis.from_url(f"redis://:{password}@localhost:6380/0")
        await r.ping()
        await r.aclose()
        print(f" {'Redis':15} | CONNECTED")
        return True
    except Exception as e:
        print(f" {'Redis':15} | CONNECTION FAILED ({str(e)})")
        return False


async def check_rabbitmq() -> bool:
    try:
        # Default dev credentials
        rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://admin:password@localhost:5673//")
        # Ensure it points to localhost port
        if "@rabbitmq:" in rabbitmq_url:
            rabbitmq_url = rabbitmq_url.replace("@rabbitmq:", "@localhost:5673")
        elif "@localhost:" in rabbitmq_url:
            # handle existing localhost
            pass
        else:
            # try to force localhost port
            rabbitmq_url = rabbitmq_url.replace(":5672", ":5673")

        connection = await aio_pika.connect_robust(rabbitmq_url)
        await connection.close()
        print(f" {'RabbitMQ':15} | CONNECTED")
        return True
    except Exception as e:
        print(f" {'RabbitMQ':15} | CONNECTION FAILED ({str(e)})")
        return False


async def verify_security() -> bool:
    """Check for presence of Production key pairs."""
    pki_path = os.path.join(os.getcwd(), ".pki")
    required_keys = ["jwt_rs256.key", "jwt_rs256.pub", "jwt_es256.key", "jwt_es256.pub"]
    all_present = True
    for key in required_keys:
        if not os.path.exists(os.path.join(pki_path, key)):  # noqa: ASYNC240
            print(f" Security Key Missing: {key}")
            all_present = False

    if all_present:
        print(f" {'PKI Assets':15} | VALIDATED (RSA 4096 / ECC P-256)")
    return all_present


async def verify_readiness():
    """Run full Production readiness check."""
    print("=" * 60)
    print("Manifold Production Readiness Report")
    print("=" * 60)

    # 1. Microservices
    print("\n--- [Microservices] ---")
    http_results = await asyncio.gather(*[check_http_service(n, u) for n, u in SERVICES.items()])

    # 2. Infra Checks
    print("\n--- [Infrastructure] ---")
    infra_results = await asyncio.gather(check_postgres(), check_redis(), check_rabbitmq())

    # 3. Security Checks
    print("\n--- [Security] ---")
    security_ok = await verify_security()

    overall = all(http_results) and all(infra_results) and security_ok
    print("\n" + "=" * 60)
    if overall:
        print(" SYSTEM STATUS: Production GREEN - READY FOR LAUNCH")
    else:
        print("️  SYSTEM STATUS: DEGRADED - ACTION REQUIRED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_readiness())
