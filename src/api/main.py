from fastapi import FastAPI, Request, Depends, HTTPException
import os
import time
import structlog
from src.shared.observability import setup_logging, logging_middleware
from src.auth.service import get_auth_service, AuthService
from src.database import get_db
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from brotli_asgi import BrotliMiddleware
from fastapi.responses import ORJSONResponse
from src.config import settings

# Initialize logging
logger = structlog.get_logger()

# Optimized event loop
try:
    import uvloop
    import asyncio
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    default_response_class=ORJSONResponse
)

# Middleware
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)

# Exception Handler
async def api_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    error_detail = str(exc)
    if settings.ENVIRONMENT != "prod": # Provide more detail in non-prod environments
        import traceback
        error_detail = traceback.format_exc()
        logger.error("api_error_detailed", error=error_detail, path=request.url.path)
    else:
        logger.error("api_error", error=str(exc), path=request.url.path)
    
    return ORJSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": error_detail if settings.ENVIRONMENT != "prod" else "An unexpected error occurred"}
    )

app.add_exception_handler(Exception, api_exception_handler)

# Dependency for Auth
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Helper to get user from state."""
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

async def get_current_active_user(user: dict = Depends(get_current_user)):
    """Helper to get active user."""
    return user

async def verify_token(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    service = AuthService(db)
    try:
        payload = service.validate_token(token)
        request.state.user = payload
        return payload
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

# Routers
from src.api.routes.auth import router as auth_router
from src.api.routes.pricing import router as pricing_router
from src.api.routes.ml import router as ml_router
from src.api.routes.users import router as users_router
from strawberry.fastapi import GraphQLRouter
from src.api.graphql.schema import schema
from src.api.middleware.security import JWTAuthenticationMiddleware

graphql_app = GraphQLRouter(schema)

app.add_middleware(JWTAuthenticationMiddleware)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy", "singularity": "achieved"}

@app.get("/")
async def root():
    return {"message": "BS-Opt Singularity API"}
