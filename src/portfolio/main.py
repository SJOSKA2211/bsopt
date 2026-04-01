import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from strawberry.fastapi import GraphQLRouter

from src.portfolio.graphql.schema import schema
from src.shared.observability import logging_middleware, setup_logging, tune_gc
from src.shared.security import opa_authorize, verify_mtls
from prometheus_fastapi_instrumentator import Instrumentator
from src.portfolio.health import get_portfolio_health

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from api.responses import MsgspecJSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize components
    setup_logging()
    tune_gc()

    # Initialize Redis for caching if needed
    from src.shared.utils.cache import init_redis_cache
    from src.shared.config import settings

    await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )

    yield
    # Shutdown logic
    from src.database import dispose_engine

    await dispose_engine()

app = FastAPI(
    title="BS-Opt Portfolio Service",
    lifespan=lifespan,
    default_response_class=MsgspecJSONResponse,
)

# Instrument for Prometheus
Instrumentator().instrument(app).expose(app)
app.middleware("http")(logging_middleware)

# Standardized Error Handling
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception) -> MsgspecJSONResponse:
    import structlog

    structlog.get_logger().error("portfolio_service_error", error=str(exc), path=request.url.path)
    return MsgspecJSONResponse(
        status_code=500,
        content={"message": "Portfolio manifold internal error", "type": "persistence_failure"},
    )

# Apply Zero Trust security dependencies
security_deps = [Depends(verify_mtls), Depends(opa_authorize("read", "portfolio"))]

graphql_app: GraphQLRouter[Any, Any] = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql", dependencies=security_deps)

@app.get("/health/liveness")
async def liveness():
    """Basic process check."""
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    """Deep check for database, redis, and risk engine sanity."""
    health_data = await get_portfolio_health()
    if health_data["status"] != "healthy":
        from fastapi import Response
        return Response(content=str(health_data), status_code=503)
    return health_data

@app.get("/health")
async def legacy_health():
    """Backward compatibility health endpoint."""
    return await get_portfolio_health()
