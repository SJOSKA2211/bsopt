import asyncio
import logging
import os
import sys
from fastapi import FastAPI
import uvicorn
from src.auth.grpc_server import serve as serve_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("auth_server")

app = FastAPI(title="Manifold Auth Service", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint for basic connectivity check."""
    return {"status": "online", "message": "Manifold Auth Service (Python gRPC Bridge) Running"}

@app.get("/health")
async def health():
    """Standard health check endpoint."""
    return {"status": "healthy", "service": "auth-service"}

async def run_servers():
    """Run both gRPC and HTTP servers concurrently."""
    grpc_port = os.getenv("GRPC_PORT", "50051")
    http_port = int(os.getenv("HTTP_PORT", 3001))
    
    logger.info(f"🚀 Starting Auth Service Mesh...")
    
    # Start gRPC in the background
    grpc_task = asyncio.create_task(serve_grpc(port=grpc_port))
    logger.info(f"📡 gRPC Server listening on port {grpc_port}")
    
    # Start HTTP server
    logger.info(f"🌐 HTTP Server (healthcheck) listening on port {http_port}")
    config = uvicorn.Config(
        app, 
        host="0.0.0.0",
        port=http_port, 
        log_level="info",
        access_log=True  # Enabled for debugging
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        logger.info("🛑 Shutting down servers...")
        grpc_task.cancel()
        try:
            await grpc_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    logger.info("!!! AUTH SERVER MANIFOLD INITIALIZING !!!")
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        logger.info("👋 Exiting...")
    except Exception as e:
        logger.exception("CRITICAL_STARTUP_FAILURE")
        sys.exit(1)
