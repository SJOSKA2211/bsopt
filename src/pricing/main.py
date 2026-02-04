from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.pricing.graphql.schema import schema, get_context
from src.shared.observability import setup_logging, logging_middleware, tune_gc
from src.pricing.quant_utils import warmup_jit
from fastapi.responses import ORJSONResponse

setup_logging()

app = FastAPI(
    title="BS-Opt Pricing Service",
    default_response_class=ORJSONResponse
)
app.middleware("http")(logging_middleware)

@app.on_event("startup")
async def startup_event():
    tune_gc()
    warmup_jit()

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}
