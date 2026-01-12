from fastapi import FastAPI, Request, Depends
from prometheus_client import make_asgi_app, Counter, Histogram
import os
import time
import structlog
from strawberry.fastapi import GraphQLRouter
from src.api.graphql.schema import schema
from src.shared.observability import setup_logging, logging_middleware
from src.shared.security import verify_mtls, opa_authorize
from contextlib import asynccontextmanager

# Initialize logging
setup_logging()
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("api_startup", version="1.0.0")
    yield
    # Shutdown logic (if any)

# Multiproc directory for Prometheus
PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/metrics")
if not os.path.exists(PROMETHEUS_MULTIPROC_DIR):
    os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)

app = FastAPI(title="BS-Opt API", lifespan=lifespan)

# Add middleware
app.middleware("http")(logging_middleware)

# Add prometheus asgi middleware to expose /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

def get_context(request: Request):
    if os.environ.get("TESTING") == "true":
        return {}  # Return empty context for tests
    return {"request": request}

# Enable GraphQL IDE only in debug or testing environments
ENABLE_IDE = os.environ.get("DEBUG", "false").lower() == "true" or os.environ.get("TESTING") == "true"

# Apply Zero Trust security dependencies:
# 1. verify_mtls ensures the request came from a trusted service (mTLS)
# 2. opa_authorize ensures the user has permission to access the options resource
security_deps = [Depends(verify_mtls), Depends(opa_authorize("read", "options"))]

graphql_app = GraphQLRouter(schema, graphql_ide=ENABLE_IDE, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql", dependencies=security_deps)

REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"])

@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
    
    return response

@app.get("/")
async def root():
    return {"message": "BS-Opt API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}