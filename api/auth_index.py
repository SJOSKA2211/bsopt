import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvloop
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes import auth_router, users_router
from src.auth.grpc_server import serve as serve_grpc
from src.config import settings

# High-performance event loop initialization
try:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except (ImportError, AttributeError):
    pass

logger = structlog.get_logger("auth_service")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Optimized Auth Service lifecycle management.
    """
    from src.database import db_manager
    from src.shared.utils.cache import init_redis_cache, close_redis_cache
    from src.auth.auth import token_blacklist

    db_manager.initialize()
    await init_redis_cache()
    
    from src.shared.utils.cache import get_redis_client
    redis_client = await get_redis_client()
    await token_blacklist.initialize(redis_client)

    yield

    from src.database import dispose_engine
    await dispose_engine()
    await close_redis_cache()
    logger.info("auth_service_shutdown_complete")

app = FastAPI(
    title="BSOPT Auth Service",
    version="2.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

# Routing Hierarchy
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(auth_router)
api_v1.include_router(users_router)
app.include_router(api_v1)

@app.get("/health")
@app.get("/health/liveness")
async def health_check():
    from src.database import health_check as db_check
    from src.shared.utils.cache import get_redis
    
    redis_ok = False
    try:
        redis = get_redis()
        redis_ok = await redis.ping()
    except Exception:
        pass

    db_res = await db_check()
    is_healthy = db_res["status"] == "healthy" and redis_ok

    return {
        "status": "healthy" if is_healthy else "degraded",
        "database": db_res,
        "redis": {"status": "healthy" if redis_ok else "unhealthy"}
    }

@app.get("/")
async def root():
    return {"status": "online", "service": "BSOPT Auth Engine"}

async def run_servers():
    """Concurrently execute gRPC and HTTP bridges."""
    grpc_port = os.getenv("GRPC_PORT", "50051")
    http_port = int(os.getenv("HTTP_PORT", 3001))

    logger.info("starting_auth_service", extra={"runtime": "uvloop", "grpc_port": grpc_port, "http_port": http_port})

    # Optional background tasks
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
