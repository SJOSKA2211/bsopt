import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from strawberry.fastapi import GraphQLRouter

from api.responses import MsgspecJSONResponse
from src.math_kernel.graphql.schema import get_context, schema
from src.math_kernel.quant_utils import warmup_jit
from src.shared.observability import (
    logging_middleware,
    setup_logging,
    tune_gc,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """High-Performance Lifespan for Pricing Service."""
    setup_logging()
    tune_gc()

    # Warmup JIT kernels to prevent first-request latency spikes
    await asyncio.to_thread(warmup_jit)

    yield

    # Shutdown logic
    from src.database import dispose_engine

    await dispose_engine()

app = FastAPI(
    title="BS-Opt Pricing Service", lifespan=lifespan, default_response_class=MsgspecJSONResponse
)

app.middleware("http")(logging_middleware)

# Standardized Error Handling
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    import structlog

    structlog.get_logger().error("pricing_service_error", error=str(exc), path=request.url.path)
    return MsgspecJSONResponse(
        status_code=500,
        content={"message": "Internal pricing engine error", "type": "computation_failure"},
    )

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    from src.database import health_check

    return {"status": "healthy", "database": health_check()}
