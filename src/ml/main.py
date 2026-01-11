from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.ml.graphql.schema import schema
from src.shared.observability import setup_logging, logging_middleware

setup_logging()

app = FastAPI(title="BS-Opt ML Service")
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}
