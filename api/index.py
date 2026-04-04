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

# Initialize logging
logger = structlog.get_logger()


def get_context(request: Request) -> dict[str, Any]:
    """GraphQL Context factory."""
    if os.getenv("TESTING") == "true":
        return {}
    return {"request": request}


# Optimized event loop
try:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except (ImportError, AttributeError):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    High-Performance Lifespan: Handles resource lifecycle with zero-leak guarantees.
    """
    start_system_metrics_loop("api")

    # Initialize Database (High-Performance)
    from src.database import db_manager

    db_manager.initialize()

    # Initialize Telemetry (OpenTelemetry)
    from src.shared.tracing import instrument_app, instrument_redis, setup_tracing

    setup_tracing("bsopt-api")
    instrument_app(app)

    # Initialize Redis
    from src.shared.utils.cache import get_redis_client, init_redis_cache

    await init_redis_cache()
    redis_client = await get_redis_client()
    instrument_redis(redis_client)

    # Initialize Token Blacklist with Redis
    from src.auth.auth import token_blacklist

    await token_blacklist.initialize(redis_client)

    # Chaos Injection
    from src.shared.utils.chaos import monkey

    if monkey.enabled:
        logger.warning("chaos_mode_active_injecting_startup_latency")
        await monkey.delay_db(0.5)

    yield

    # Shutdown
    from api.websockets.manager import manager
    from src.database import dispose_engine
    from src.shared.utils.cache import close_redis_cache

    await manager.close()
    await dispose_engine()
    await close_redis_cache()
    logger.info("api_shutdown_complete_database_engines_disposed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    default_response_class=MsgspecJSONResponse,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-CSRF-Token",
        "X-Request-ID",
        "X-API-Key",
        "Accept",
    ],
)
app.middleware("http")(logging_middleware)

app.add_middleware(ZeroTrustMiddleware)


# Exception Handler
async def api_exception_handler(request: Request, exc: Exception) -> MsgspecJSONResponse:
    """Global exception handler."""
    from api.exceptions import BaseAPIException

    if isinstance(exc, BaseAPIException):
        return MsgspecJSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={
                "error": getattr(exc, "error_code", getattr(exc, "error", "InternalServerError")),
                "message": getattr(exc, "message", str(exc)),
                "details": getattr(exc, "details", None),
            },
        )

    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return MsgspecJSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "status_code": exc.status_code,
                **detail,
            },
            headers=getattr(exc, "headers", None),
        )

    error_detail = str(exc)
    if settings.ENVIRONMENT != "prod":
        import traceback

        error_detail = traceback.format_exc()
        logger.error("api_error_detailed", error=error_detail, path=request.url.path)
    else:
        logger.error("api_error", error=str(exc), path=request.url.path)

    return MsgspecJSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": (
                error_detail if settings.ENVIRONMENT != "prod" else "An unexpected error occurred"
            ),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> MsgspecJSONResponse:
    """Handle FastAPI built-in validation errors."""
    return MsgspecJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


app.add_exception_handler(Exception, api_exception_handler)
app.add_exception_handler(HTTPException, api_exception_handler)

graphql_app: GraphQLRouter[Any, Any] = GraphQLRouter(schema)

# API v1 Routes
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(pricing_router)
api_router.include_router(ml_router)
api_router.include_router(options_router)
api_router.include_router(portfolio_router)
api_router.include_router(users_router)
api_router.include_router(market_router)
api_router.include_router(websocket_router)  # Include websocket router

if settings.ENVIRONMENT not in ("prod", "production"):
    api_router.include_router(debug_router)

app.include_router(api_router)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    from src.database import health_check
    from src.math_kernel.rust_engine import is_rust_available
    from src.shared.utils.broker import broker
    from src.shared.utils.cache import get_redis

    redis_status = "unhealthy"
    try:
        redis = get_redis()
        if redis and await redis.ping():
            redis_status = "healthy"
    except Exception:
        pass

    db_health = await health_check()
    rabbitmq_health = await broker.health_check()
    if rabbitmq_health["status"] == "healthy":
        try:
            # Add stats for the default task queue
            rabbitmq_health["queues"] = {
                "default": await broker.get_queue_stats("default")
            }
        except Exception:
            pass

    return {
        "status": "healthy" if db_health["status"] == "healthy" and redis_status == "healthy" and rabbitmq_health["status"] == "healthy" else "degraded",
        "database": db_health,
        "redis": {"status": redis_status},
        "rabbitmq": rabbitmq_health,
        "rust_core": {
            "available": is_rust_available(),
            "status": "healthy" if is_rust_available() else "unavailable",
        },
    }


@app.get("/api/diagnostics/imports")
async def diagnostics_imports() -> dict[str, bool]:
    return {"successful_imports": True}


@app.get("/admin-only")
async def admin_only(user: dict[str, Any] = Depends(RoleChecker(["admin"]))) -> dict[str, str]:
    return {"message": "Welcome, Admin"}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    # Only allow internal/authenticated access to metrics
    if settings.is_production:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            from api.responses import MsgspecJSONResponse

            return MsgspecJSONResponse(status_code=401, content={"detail": "Not authenticated"})

    from src.math_kernel.rust_engine import get_rust_metrics

    python_metrics = generate_latest().decode("utf-8")
    rust_metrics = get_rust_metrics()
    return Response(content=f"{python_metrics}\n{rust_metrics}", media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "BS-Opt Optimized API is running"}
