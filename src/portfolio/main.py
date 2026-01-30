from fastapi import FastAPI, Depends
from fastapi.responses import ORJSONResponse
from strawberry.fastapi import GraphQLRouter
from src.portfolio.graphql.schema import schema
from src.shared.observability import setup_logging, logging_middleware, tune_gc
from src.shared.security import verify_mtls, opa_authorize
from contextlib import asynccontextmanager
import os
import asyncio

# Optimized event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components
    setup_logging()
    tune_gc()
    
    # Initialize Redis for caching if needed
    from src.utils.cache import init_redis_cache
    from src.config import settings
    await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )
    
    yield
    # Shutdown logic

app = FastAPI(
    title="BS-Opt Portfolio Service",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)
app.middleware("http")(logging_middleware)

# Apply Zero Trust security dependencies
security_deps = [Depends(verify_mtls), Depends(opa_authorize("read", "portfolio"))]

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql", dependencies=security_deps)

@app.get("/health")
async def health():
    return {"status": "healthy"}
