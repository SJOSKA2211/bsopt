# Design: API Health Revamp (Approach 1)

## 1. Problem Statement
The current API health check is a basic heartbeat. We need a more comprehensive "Engine" that reports internal health, upstream dependency status, and performance metrics (latency, error rates).

## 2. Approach: Integrated API Sentinel & Dependency Awareness
We will enhance the API's internal health logic and integrate it with the `system_sentinel.py`.

## 3. Architecture Changes

### 3.1. API Health Logic Enhancement
- **Dependency Health**: The `/health` endpoint will now check connections to Redis, PgBouncer, and ML Inference.
- **Metrics Endpoint**: Expose a `/metrics` endpoint (Prometheus format) for granular performance tracking.
- **Background Health Task**: A background task within the API will periodically refresh health status to avoid blocking the health endpoint on slow dependencies.

### 3.2. System Sentinel Integration
- **API Check**: Add `check_api` to `system_sentinel.py` to query `http://api:8000/health`.
- **Metrics Reporting**: Sentinel will report average latency and error rates from the API.

## 4. Implementation Details

### API Health Check Extension
```python
# src/api/routes/health.py
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dependencies": {
            "database": await db_check(),
            "redis": await redis_check(),
            "ml_inference": await ml_check()
        }
    }
```

## 5. Success Criteria
- Sentinel reports API health and dependency status.
- API service is restarted automatically if dependencies are sustained unhealthy.
- Performance metrics (latency) are visible in the sentinel output.
