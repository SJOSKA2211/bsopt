import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvloop
from brotli_asgi import BrotliMiddleware
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from api.graphql.schema import schema
from api.middleware.fused import ZeroTrustMiddleware
from api.responses import MsgspecJSONResponse
from api.routes import (
    auth_router,
    debug_router,
    market_router,
    ml_router,
    options_router,
    portfolio_router,
    pricing_router,
    users_router,
    websocket_router,
)
from src.auth.auth import RoleChecker
from src.config import settings
from src.shared.observability import logging_middleware, start_system_metrics_loop

# High-performance event loop initialization
try:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except (ImportError, AttributeError):
    pass

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Optimized application lifecycle management.
    """
    start_system_metrics_loop("api")

    # Initialize core services
    from src.database import db_manager
    from src.shared.tracing import instrument_app, instrument_redis, setup_tracing
    from src.shared.utils.cache import get_redis_client, init_redis_cache
    from src.auth.auth import token_blacklist

    db_manager.initialize()
    setup_tracing("bsopt-api")
    instrument_app(app)

    await init_redis_cache()
    redis_client = await get_redis_client()
    instrument_redis(redis_client)
    await token_blacklist.initialize(redis_client)

    yield

    # Clean shutdown
    from api.websockets.manager import manager
    from src.database import dispose_engine
    from src.shared.utils.cache import close_redis_cache

    await manager.close()
    await dispose_engine()
    await close_redis_cache()
    logger.info("api_shutdown_complete")

app = FastAPI(
    title=settings.PROJECT_NAME,
    default_response_class=MsgspecJSONResponse,
    lifespan=lifespan,
)

# Middleware Pipeline
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-CSRF-Token", "X-Request-ID", "X-API-Key", "Accept"],
)
app.middleware("http")(logging_middleware)
app.add_middleware(ZeroTrustMiddleware)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> MsgspecJSONResponse:
    """Consolidated global exception handling."""
    from api.exceptions import BaseAPIException

    if isinstance(exc, BaseAPIException):
        return MsgspecJSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={
                "error": getattr(exc, "error_code", "InternalError"),
                "message": getattr(exc, "message", str(exc)),
                "details": getattr(exc, "details", None),
            },
        )

    if isinstance(exc, HTTPException):
        return MsgspecJSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "message": str(exc.detail)},
        )

    logger.exception("api_unhandled_exception", path=request.url.path)
    return MsgspecJSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc) if settings.DEVELOPMENT else "Unexpected error"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> MsgspecJSONResponse:
    return MsgspecJSONResponse(status_code=422, content={"detail": exc.errors()})

# Routing Hierarchy
api_v1 = APIRouter(prefix="/api/v1")
for router in [auth_router, pricing_router, ml_router, options_router, portfolio_router, users_router, market_router, websocket_router]:
    api_v1.include_router(router)

if settings.DEVELOPMENT:
    api_v1.include_router(debug_router)

app.include_router(api_v1)
app.include_router(GraphQLRouter(schema), prefix="/graphql")

@app.get("/health")
@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    """Unified system health telemetry."""
    from src.database import health_check as db_check
    from src.math_kernel.rust_engine import is_rust_available
    from src.shared.utils.broker import broker
    from src.shared.utils.cache import get_redis, get_redis_pool_stats

    redis_ok, redis_metrics = False, {}
    try:
        redis = get_redis()
        redis_ok = await redis.ping()
        redis_metrics = await get_redis_pool_stats()
    except Exception:
        pass

    db_res = await db_check()
    mq_res = await broker.health_check()

    is_healthy = all([db_res["status"] == "healthy", redis_ok, mq_res["status"] == "healthy"])

    return {
        "status": "healthy" if is_healthy else "degraded",
        "database": db_res,
        "redis": {"status": "healthy" if redis_ok else "unhealthy", "metrics": redis_metrics},
        "rabbitmq": mq_res,
        "rust_core": {"available": is_rust_available(), "status": "healthy" if is_rust_available() else "unavailable"},
    }

@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """System and Rust-core metrics."""
    if settings.is_production:
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            return MsgspecJSONResponse(status_code=401, content={"detail": "Unauthorized"})

    from src.math_kernel.rust_engine import get_rust_metrics
    return Response(content=f"{generate_latest().decode()}\n{get_rust_metrics()}", media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "BSOPT API Active"}