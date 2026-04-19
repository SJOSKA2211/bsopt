"""FastAPI application entry point."""

import logging

import grpc
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from api.routes.market_data import router as market_data_router
from api.routes.ml import router as ml_router
from api.routes.portfolio import router as portfolio_router
from api.routes.trade import router as trade_router

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="BSOPT API",
    description="Backend API for the Black-Scholes Advanced Option Pricing Platform",
    version="2.5.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# --- Middlewares ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(market_data_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(portfolio_router, prefix="/api/v1")
app.include_router(trade_router, prefix="/api/v1")

# --- Exception Handlers ---
@app.exception_handler(grpc.RpcError)
async def grpc_exception_handler(request: Request, exc: grpc.RpcError):
    """Custom handler for gRPC RpcErrors."""
    logger.error(f"gRPC RpcError: Code={exc.code()}, Details={exc.details()}")

    if exc.code() == grpc.StatusCode.UNAUTHENTICATED:
        status_code = status.HTTP_401_UNAUTHORIZED
    elif exc.code() == grpc.StatusCode.PERMISSION_DENIED:
        status_code = status.HTTP_403_FORBIDDEN
    elif exc.code() == grpc.StatusCode.NOT_FOUND:
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code() == grpc.StatusCode.UNAVAILABLE:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    detail = exc.details() if exc.details() else "gRPC service error"
    return HTTPException(status_code=status_code, detail=detail)

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "message": "API is healthy"}

# --- Root Endpoint ---
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for the API."""
    return {"message": "Welcome to the BSOPT API!"}
