import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from strawberry.fastapi import GraphQLRouter

from src.portfolio.graphql.schema import schema
from src.shared.observability import logging_middleware, setup_logging, tune_gc
from src.shared.security import opa_authorize, verify_mtls

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from services.api.responses import MsgspecJSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize components
    setup_logging()
    tune_gc()

    # Initialize Redis for caching if needed
    from src.shared.cache import init_redis_cache
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

@app.get("/health")
async def health() -> dict[str, Any]:
    from src.database import health_check

    return {"status": "healthy", "database": health_check()}
