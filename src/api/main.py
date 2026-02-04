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

# Dependency for Auth
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

app.include_router(auth_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(ml_router, prefix="/api/v1", dependencies=[Depends(verify_token)])

@app.get("/health")
async def health():
    return {"status": "healthy", "singularity": "achieved"}

@app.get("/")
async def root():
    return {"message": "BS-Opt Singularity API"}
