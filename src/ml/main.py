import asyncio
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.ml.graphql.schema import schema, get_context
from src.shared.observability import logging_middleware, setup_logging

# Optimized event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

setup_logging()

from fastapi.responses import ORJSONResponse

app = FastAPI(
    title="BS-Opt ML Service",
    default_response_class=ORJSONResponse
)
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}
