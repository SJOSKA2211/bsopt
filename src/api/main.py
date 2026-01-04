from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app, Counter, Histogram
import os
import time

# Multiproc directory for Prometheus
PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/metrics")
if not os.path.exists(PROMETHEUS_MULTIPROC_DIR):
    os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)

app = FastAPI(title="BS-Opt API")

# Add prometheus asgi middleware to expose /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"])

@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    
    # We use request.scope.get('path') to avoid including dynamic path parameters in the label if possible
    # but for now we'll use simple request.url.path
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