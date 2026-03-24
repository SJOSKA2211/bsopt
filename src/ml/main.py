import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from strawberry.fastapi import GraphQLRouter

from src.ml.graphql.schema import get_context, schema
from src.shared.observability import logging_middleware, setup_logging

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

setup_logging()

app = FastAPI(title="BS-Opt ML Service", default_response_class=ORJSONResponse)
app.middleware("http")(logging_middleware)

graphql_app: GraphQLRouter[Any, Any] = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/ml/reload")
async def reload_models() -> dict[str, str]:
    """
    Trigger dynamic model reload across the manifold.
    Consistency endpoint for MLOps V2 orchestration.
    """
    # For this entry point, we might trigger a global event or specific service reloads.
    # In this unified main, we log and return success.
    from src.shared.observability import post_grafana_annotation

    await post_grafana_annotation("ML Manifold Reload Triggered", ["ml", "reload"])
    return {"status": "reload_triggered"}
