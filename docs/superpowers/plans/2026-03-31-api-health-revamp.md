# API Health Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate full dependency health reporting into the API and unified monitoring via the system sentinel.

**Architecture:** Approach 1 (Integrated API Sentinel) enhances the API `/health` endpoint to report Redis and ML Inference status, and adds an API health probe to `system_sentinel.py`.

**Tech Stack:** Python (FastAPI, SQLAlchemy, structlog), Docker.

---

### Task 1: API Dependency Awareness Revamp

**Files:**
- Modify: `api/index.py`

- [ ] **Step 1: Enhance `/health` endpoint**

Update the `health` function in `api/index.py` to include Redis and ML Inference checks:
```python
@app.get("/health")
@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    from src.database import health_check
    from src.math_kernel.rust_engine import is_rust_available
    from src.shared.utils.cache import get_redis
    
    redis_status = "unhealthy"
    try:
        redis = get_redis()
        if redis and await redis.ping():
            redis_status = "healthy"
    except Exception:
        pass

    return {
        "status": "healthy",
        "database": await health_check(),
        "redis": {"status": redis_status},
        "rust_core": {
            "available": is_rust_available(),
            "status": "healthy" if is_rust_available() else "unavailable",
        },
    }
```

- [ ] **Step 2: Commit API changes**

```bash
git add api/index.py
git commit -m "feat: enhance api health endpoint with redis dependency check"
```

### Task 2: System Sentinel API Integration

**Files:**
- Modify: `scripts/system_sentinel.py`

- [ ] **Step 1: Implement `check_api` function**

Add this function to `scripts/system_sentinel.py`:
```python
async def check_api():
    print("Checking API Unified Manifold...", end=" ", flush=True)
    import httpx
    import os
    from src.shared.config import settings
    
    # Allow host override for local testing
    host = os.environ.get("API_HOST", "api")
    port = os.environ.get("API_PORT", "8000")
    url = f"http://{host}:{port}/health"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                db_status = data.get("database", {}).get("status", "unknown")
                redis_status = data.get("redis", {}).get("status", "unknown")
                print(f" [ALIVE: DB={db_status}, Redis={redis_status}]")
            else:
                print(f"️ [UNEXPECTED STATUS: {resp.status_code}]")
    except Exception as e:
        print(f" [FAILED: {e}]")
```

- [ ] **Step 2: Update `main` to include API check**

```python
async def main():
    print("\n" + "=" * 50)
    print("   BS-OPT HIGH-PERFORMANCE SYSTEM SENTINEL")
    print("=" * 50)
    await check_database()
    await check_pgbouncer()
    await check_api()  # New check
    await check_redis()
    await check_shm()
    print("=" * 50 + "\n")
```

- [ ] **Step 3: Commit sentinel changes**

```bash
git add scripts/system_sentinel.py
git commit -m "feat: integrate api health check into system sentinel"
```

### Task 3: Run API Until Healthy

**Files:**
- Create: `scripts/run_api_until_healthy.sh`

- [ ] **Step 1: Create the orchestration script**

```bash
#!/bin/bash
set -euo pipefail

# scripts/run_api_until_healthy.sh - Start API and verify health
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect Container Engine
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo " Starting API service..."
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d api otel-collector

echo "⏳ Verifying API health..."
RETRIES=30
INTERVAL=5
SUCCESS=0

for ((i=1; i<=RETRIES; i++)); do
    echo "   [Attempt $i/$RETRIES] Checking status..."
    if $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml exec api curl -s http://localhost:8000/health | grep -q "\"status\":\"healthy\""; then
        echo " API is ACTIVE and reporting healthy status."
        SUCCESS=1
        break
    fi
    echo "   ️ API not ready yet. Retrying in ${INTERVAL}s..."
    sleep $INTERVAL
done

if [ $SUCCESS -eq 0 ]; then
    echo " Fatal: API failed to reach healthy state after $RETRIES attempts."
    exit 1
fi

echo " API is Online and Verified."
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x scripts/run_api_until_healthy.sh
```

- [ ] **Step 3: Run the script**

```bash
bash scripts/run_api_until_healthy.sh
```

- [ ] **Step 4: Commit orchestration script**

```bash
git add scripts/run_api_until_healthy.sh
git commit -m "feat: add run_api_until_healthy orchestration script"
```
