from fastapi import FastAPI, Depends
from strawberry.fastapi import GraphQLRouter
from src.portfolio.graphql.schema import schema
from src.shared.observability import setup_logging, logging_middleware
from src.shared.security import verify_mtls, opa_authorize
import os

setup_logging()

app = FastAPI(title="BS-Opt Portfolio Service")
app.middleware("http")(logging_middleware)

# Apply Zero Trust security dependencies
security_deps = [Depends(verify_mtls), Depends(opa_authorize("read", "portfolio"))]

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql", dependencies=security_deps)

@app.get("/health")
async def health():
    return {"status": "healthy"}
