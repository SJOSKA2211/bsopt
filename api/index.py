"""FastAPI application entry point."""

import logging
from typing import Any

import grpc
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
async def grpc_exception_handler(_: Request, exc: grpc.RpcError) -> JSONResponse:
    """Handle gRPC RpcErrors by converting them to FastAPI HTTPExceptions."""
    logger.error("gRPC RpcError: Code=%s, Details=%s", exc.code(), exc.details())

    status_map = {
        grpc.StatusCode.UNAUTHENTICATED: status.HTTP_401_UNAUTHORIZED,
        grpc.StatusCode.PERMISSION_DENIED: status.HTTP_403_FORBIDDEN,
        grpc.StatusCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
        grpc.StatusCode.UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    
    status_code = status_map.get(exc.code(), status.HTTP_500_INTERNAL_SERVER_ERROR)
    detail = exc.details() or "gRPC service error"
    code = exc.code().name
    
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": detail},
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Provide a basic health check endpoint."""
    return {"status": "ok", "message": "API is healthy"}

# --- Root Endpoint ---
@app.get("/")
async def root() -> dict[str, str]:
    """Provide a root endpoint for the API."""
    return {"message": "Welcome to the BSOPT API!"}
