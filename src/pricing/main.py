from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.pricing.graphql.schema import schema, get_context

setup_logging()

from fastapi.responses import ORJSONResponse

app = FastAPI(
    title="BS-Opt Pricing Service",
    default_response_class=ORJSONResponse
)
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}
