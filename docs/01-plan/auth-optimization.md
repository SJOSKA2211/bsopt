# Plan: Run, Report, and Optimize Auth Engine

## Objective
Start the Manifold Auth service, verify its health, report on its status, and implement architectural optimizations to improve performance and reduce database load.

## Background & Motivation
The current Auth engine relies heavily on frequent database calls for user information and API key validation over gRPC. By running the service and monitoring its health, we establish a baseline. Subsequent optimizations (like caching) will significantly enhance throughput and decrease latency.

## Scope & Impact
- **Affected Services**: `src/auth/auth_server.py`, `src/auth/grpc_server.py`
- **Impact**: Improves token validation speed, reduces DB load on `/health` and gRPC endpoints.

## Proposed Solution

### Phase 1: Execution and Health Reporting
1. Launch the authentication service in the background using `scripts/start_auth.sh`.
2. Run `scripts/run_auth_healthy.py` to monitor and report the engine's HTTP and gRPC health status.
3. Fetch the detailed health report from `/health/readiness`.

### Phase 2: Revamp & Optimization
1. **Implement Caching**: Introduce a caching layer (e.g., in-memory async LRU cache) in `src/auth/grpc_server.py` for:
   - `GetUserInfo`: Cache user profiles based on `user_id`.
   - `ValidateAPIKey`: Cache hashed API key validation results.
2. **Database Session Management**: Optimize the DB session lifecycle in gRPC handlers to prevent connection overhead (reusing sessions or utilizing a connection pool effectively).
3. **Verify Optimizations**: Rerun the health checks and validation tests to ensure the newly optimized engine is stable and performs better.

## Alternatives Considered
- *Full Redis Cache*: More robust but adds dependency complexity if Redis isn't strictly required for auth specifically (though it's in the stack). We will check if `redis` is available via `src.database` or use a local async cache.

## Verification & Testing
- The output of `run_auth_healthy.py` must print "AUTH SERVICE MANIFOLD IS FULLY HEALTHY".
- Unit/integration tests (if existing) pass after caching logic is introduced.
