from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter
import os

app = FastAPI(title="BS-Opt API")

# Add prometheus asgi middleware to expose /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint"])

@app.get("/")
async def root():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    return {"message": "BS-Opt API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
