from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time
import orjson
import asyncio
from typing import Any, Dict

class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, producer: Any = None, topic: str = "audit-logs"):
        super().__init__(app)
        self.producer = producer
        self.topic = topic

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Capture audit information
        # user_id is expected to be set in request.state.user by the auth middleware
        user = getattr(request.state, "user", {})
        if hasattr(user, "id"):
            user_id = str(user.id)
        elif isinstance(user, dict):
            user_id = user.get("sub", "anonymous")
        else:
            user_id = "anonymous"
        
        audit_payload = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "user_id": user_id,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "latency_ms": (time.time() - start_time) * 1000
        }
        
        # Get producer from app state if not provided at init
        producer = self.producer or getattr(request.app.state, "audit_producer", None)
        
        if producer:
            # Asynchronously send to Kafka using orjson for speed
            try:
                producer.produce(
                    self.topic, 
                    orjson.dumps(audit_payload)
                )
                # poll(0) serves delivery callbacks without blocking
                producer.poll(0)
            except Exception:
                # Fail silently to avoid interrupting the request lifecycle
                pass
            
        return response
