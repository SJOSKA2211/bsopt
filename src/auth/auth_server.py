import asyncio
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.auth.grpc_server import serve as serve_grpc
from src.auth.health import get_overall_health

# Attempt to use uvloop for high performance
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("auth_server")

app = FastAPI(title="Manifold Auth Service", version="1.0.0", default_response_class=ORJSONResponse)

# Instrument for Prometheus
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    """Root endpoint for basic connectivity check."""
    return {"status": "online", "message": "Manifold Auth Service (Python gRPC Bridge) Running"}


@app.get("/health/liveness")
async def liveness():
    """Basic process check."""
    return {"status": "alive"}


@app.get("/health/readiness")
async def readiness():
    """Deep check for database and vault connectivity."""
    health_data = await get_overall_health()
    if health_data["status"] != "healthy":
        from fastapi import Response

        return Response(content=str(health_data), status_code=503)
    return health_data


@app.get("/health")
async def legacy_health():
    """Backward compatibility health endpoint."""
    return await get_overall_health()


async def run_servers():
    """Run both gRPC and HTTP servers concurrently."""
    grpc_port = os.getenv("GRPC_PORT", "50051")
    http_port = int(os.getenv("HTTP_PORT", 3001))

    logger.info("🚀 Starting Auth Service Mesh with uvloop...")

    # Start background tasks
    from src.auth.tasks import flush_api_key_usage_loop

    tasks = [
        asyncio.create_task(serve_grpc(port=grpc_port)),
        asyncio.create_task(flush_api_key_usage_loop()),
    ]

    logger.info(f"📡 gRPC Server listening on port {grpc_port}")
    logger.info(f"🌐 HTTP Server (healthcheck) listening on port {http_port}")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=http_port,
        log_level="info",
        access_log=False,  # Optimization: Disable access logs in production
        loop="uvloop" if "uvloop" in sys.modules else "auto",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        logger.info("🛑 Shutting down servers and background tasks...")
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    logger.info("!!! AUTH SERVER MANIFOLD INITIALIZING !!!")
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        logger.info("👋 Exiting...")
    except Exception:
        logger.exception("CRITICAL_STARTUP_FAILURE")
        sys.exit(1)
