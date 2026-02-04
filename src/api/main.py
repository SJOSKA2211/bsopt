from fastapi import FastAPI, Request, Depends, HTTPException
from prometheus_client import make_asgi_app, Counter, Histogram
import os
import time
import structlog
from src.shared.observability import setup_logging, logging_middleware
from src.shared.tracing import setup_tracing, instrument_app
from src.shared.security import verify_mtls, opa_authorize, get_zero_trust_deps
from src.auth.security import verify_token, RoleChecker
from src.audit.middleware import AuditMiddleware
from starlette.responses import JSONResponse
from starlette import status
from src.api.exceptions import BaseAPIException
from contextlib import asynccontextmanager
from src.utils.lazy_import import get_import_stats, preload_modules
import asyncio
from src.config import settings

# Deferred imports for startup speed
# from strawberry.fastapi import GraphQLRouter
# from src.api.graphql.schema import schema
# from confluent_kafka import Producer

# Optimized event loop for Linux
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Initialize logging
logger = structlog.get_logger()

from fastapi.responses import HTMLResponse

app = FastAPI(title=settings.PROJECT_NAME)

@app.get("/reference", include_in_schema=False)
async def scalar_reference():
    """🚀 SINGULARITY: High-fidelity Scalar API Oracle."""
    return HTMLResponse(content=f"""
        <!doctype html>
        <html>
          <head>
            <title>BS-OPT God-Mode Oracle</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
          </head>
          <body>
            <script id="api-reference" data-url="/openapi.json"></script>
            <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
          </body>
        </html>
    """)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logging
    setup_logging()
    
    # Initialize Tracing
    setup_tracing("bsopt-api")
    instrument_app(app)
    
    # Tune GC for performance
    from src.shared.observability import tune_gc, tune_worker_resources
    tune_gc()
    tune_worker_resources()
    
    # SOTA: Initialize Ray for distributed compute
    import ray
    if not ray.is_initialized():
        # Limit resources to prevent OOM on smaller systems
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, object_store_memory=10**9) # 1GB
    
    logger.info("application_startup_begin")

    # Initialize Redis Cache
    from src.utils.cache import init_redis_cache
    redis_conn = await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )

    # Initialize Circuit Breakers
    from src.utils.circuit_breaker import initialize_circuits
    await initialize_circuits(redis_conn)

    # Preload critical modules during startup to reduce first-request latency.
    from src.ml import preload_critical_modules as ml_preload
    from src.pricing import preload_classical_pricers as pricing_preload
    from src.utils.cache import warm_cache
    
    ml_preload()
    pricing_preload()
    
    # Pre-warm cache with common scenarios
    try:
        await warm_cache()
        logger.info("cache_warming_complete")
    except Exception as e:
        logger.warning(f"cache_warming_failed: {e}")

    stats = get_import_stats()
    logger.info(
        "preload_complete",
        modules_loaded=stats['successful_imports'],
        total_time=f"{stats['total_import_time']:.2f}s"
    )
    
    kafka_servers = os.environ.get("KAFKA_SERVERS", "localhost:9092")
    logger.info("api_startup", version="1.0.0", kafka_servers=kafka_servers)
    yield
    # Shutdown logic: Flush pending audit logs and close connections
    logger.info("api_shutdown")
    if hasattr(app.state, 'audit_producer') and app.state.audit_producer:
        app.state.audit_producer.flush(timeout=5)
    
    from src.utils.http_client import HttpClientManager
    from src.utils.cache import close_redis_cache
    
    await HttpClientManager.close()
    await close_redis_cache()
    
    # SOTA: Shutdown Ray
    import ray
    if ray.is_initialized():
        ray.shutdown()
        
    logger.info("connections_closed")

# Multiproc directory for Prometheus
PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/metrics")
if not os.path.exists(PROMETHEUS_MULTIPROC_DIR):
    os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
from src.api.routes.auth import router as auth_router
from src.api.routes.pricing import router as pricing_router
from src.api.routes.users import router as users_router
from src.api.routes.ml import router as ml_router
from src.api.routes.websocket import router as websocket_router

from src.api.middleware.security import (
    SecurityHeadersMiddleware,
    CSRFMiddleware,
    IPBlockMiddleware,
    InputSanitizationMiddleware
)
from fastapi.middleware.cors import CORSMiddleware

from brotli_asgi import BrotliMiddleware
from fastapi.responses import ORJSONResponse

app = FastAPI(
    title="BS-Opt API", 
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add Brotli compression for large responses (Superior to GZip for JSON)
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)

# Add middleware (Order matters: outermost first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CSRFMiddleware)
app.add_middleware(IPBlockMiddleware)
app.add_middleware(InputSanitizationMiddleware)
app.add_middleware(AuditMiddleware)
app.middleware("http")(logging_middleware)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(websocket_router)

# Add prometheus asgi middleware to expose /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

def get_context(request: Request):
    return {"request": request}

# Enable GraphQL IDE only in debug or testing environments
ENABLE_IDE = os.environ.get("DEBUG", "false").lower() == "true"

# Apply Zero Trust security dependencies:
# Centralized policy management ensures consistent security across all endpoints.
# 1. verify_token ensures the user is authenticated via Keycloak
# 2. verify_mtls ensures the request came from a trusted service (mTLS)
# 3. opa_authorize ensures the user has permission to access the options resource
security_deps = get_zero_trust_deps("read", "options")

def _init_graphql():
    from strawberry.fastapi import GraphQLRouter
    from src.api.graphql.schema import schema
    return GraphQLRouter(schema, graphql_ide=ENABLE_IDE, context_getter=get_context)

app.include_router(_init_graphql(), prefix="/graphql", dependencies=security_deps)

REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"])

@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add response time header
    response.headers["X-Response-Time"] = str(process_time)
    
    # Add rate limit headers if set by rate_limit dependency
    if hasattr(request.state, "rate_limit_limit"):
        response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
    if hasattr(request.state, "rate_limit_remaining"):
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
    if hasattr(request.state, "rate_limit_reset"):
        response.headers["X-RateLimit-Reset"] = str(request.state.rate_limit_reset)
    
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)
    
    return response

@app.get("/")
async def root():
    return {"message": "BS-Opt API is running"}

@app.get("/health")
@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}

@app.get("/admin-only", dependencies=[Depends(RoleChecker(allowed_roles=["admin"]))])
async def admin_only():
    """Endpoint accessible only by users with the 'admin' role."""
    return {"message": "Welcome, Admin"}

@app.get("/api/diagnostics/imports", dependencies=[Depends(RoleChecker(allowed_roles=["admin"]))])
async def get_import_diagnostics():
    """ Endpoint to inspect lazy import performance. Only available in non-prod. """
    if settings.ENVIRONMENT == "prod":
        raise HTTPException(status_code=404, detail="Endpoint not available in production")
        
    stats = get_import_stats()
    return {
        "successful_imports": stats['successful_imports'],
        "failed_imports": stats['failed_imports'],
        "total_import_time_seconds": stats['total_import_time'],
        "slowest_imports": [
            {"module": k, "duration_ms": v * 1000} for k, v in stats['slowest_imports']
        ],
        "failures": stats['failures']
    }

@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "message": exc.message,
            "details": exc.details,
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
        },
    )