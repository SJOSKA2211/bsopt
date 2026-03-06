from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from strawberry.fastapi import GraphQLRouter

from src.pricing.graphql.schema import get_context, schema
from src.pricing.quant_utils import warmup_jit
from src.shared.observability import (
    logging_middleware,
    setup_logging,
    tune_gc,
)

setup_logging()

app = FastAPI(title="BS-Opt Pricing Service", default_response_class=ORJSONResponse)
app.middleware("http")(logging_middleware)


@app.on_event("startup")
async def startup_event():
    tune_gc()
    warmup_jit()


@app.on_event("shutdown")
async def shutdown_event():
    from src.database import dispose_engine

    await dispose_engine()


graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
async def health():
    from src.database import health_check

    return {"status": "healthy", "database": health_check()}
