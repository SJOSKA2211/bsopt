import asyncio
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.auth.grpc_server import serve as serve_grpc
from src.auth.health import get_overall_health

# Initialize high-performance event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = logging.getLogger("auth_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(title="BSOPT Auth Service", version="2.0.0", default_response_class=ORJSONResponse)
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"status": "online", "service": "BSOPT Auth Engine"}

@app.get("/health/liveness")
async def liveness():
    return {"status": "alive"}

@app.get("/health/readiness")
@app.get("/health")
async def health_check():
    """Deep health inspection for DB and Vault."""
    health_data = await get_overall_health()
    if health_data.get("status") != "healthy":
        return Response(content=str(health_data), status_code=503)
    return health_data

async def run_servers():
    """Concurrently execute gRPC and HTTP bridges."""
    grpc_port = os.getenv("GRPC_PORT", "50051")
    http_port = int(os.getenv("HTTP_PORT", 3001))

    logger.info("starting_auth_service", extra={"runtime": "uvloop", "grpc_port": grpc_port, "http_port": http_port})

    from src.auth.tasks import flush_api_key_usage_loop

    tasks = [
        asyncio.create_task(serve_grpc(port=grpc_port)),
        asyncio.create_task(flush_api_key_usage_loop()),
    ]

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=http_port,
        log_level="info",
        access_log=False,
        loop="uvloop" if "uvloop" in sys.modules else "auto",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        logger.info("shutting_down_auth_service")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(run_servers())
    except Exception:
        logger.exception("CRITICAL_STARTUP_FAILURE")
        sys.exit(1)