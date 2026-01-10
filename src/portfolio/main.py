from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.portfolio.graphql.schema import schema

app = FastAPI(title="BS-Opt Portfolio Service")
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
