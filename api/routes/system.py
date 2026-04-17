import logging
import os
from typing import Any

import anyio
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.jwt_validator import require_tier
from api.responses import MsgspecJSONResponse
from api.schemas.common import DataResponse, DataResponseStruct
from src.database import crud, get_async_db
from src.shared.shm_mesh import SharedMemoryRingBuffer
from src.shared.utils.circuit_breaker import db_circuit, pricing_circuit

router = APIRouter(prefix="/system", tags=["System"], default_response_class=MsgspecJSONResponse)
logger = logging.getLogger(__name__)

# Global Probe Cache
_shm_probe = None


@router.get(
    "/health/deep",
    response_model=DataResponse[dict[str, Any]],
    dependencies=[Depends(require_tier(["admin", "enterprise"]))],
)
async def get_deep_health():
    """High-fidelity stack probe with cached connections."""
    global _shm_probe
    health = {"status": "operational", "probes": {}}

    # 1. SHM Mesh Probe (Cached)
    try:
        if _shm_probe is None:
            _shm_probe = SharedMemoryRingBuffer(create=False)

        # Lock-free head check via pre-mapped buffer
        import struct

        head = struct.unpack("q", _shm_probe.buf[:8])[0]
        health["probes"]["shm_mesh"] = {"status": "connected", "head": head}
    except Exception as e:
        logger.error("shm_mesh_probe_failed", error=str(e))
        health["probes"]["shm_mesh"] = {"status": "unavailable"}
        health["status"] = "degraded"
        _shm_probe = None  # Reset on failure

    # 2. WASM OPA Probe
    wasm_path = "policies/authz.wasm"
    try:
        exists = await anyio.to_thread.run_sync(os.path.exists, wasm_path)
    except Exception:
        exists = False

    health["probes"]["wasm_opa"] = {
        "status": "verified" if exists else "missing",
        "path": wasm_path,
    }

    # 4. Redis Probe
    from src.shared.utils.cache import get_redis

    redis = get_redis()
    if redis:
        try:
            await redis.ping()
            health["probes"]["redis"] = {"status": "connected"}
        except Exception as e:
            health["probes"]["redis"] = {"status": "error", "message": str(e)}
            health["status"] = "degraded"
    else:
        health["probes"]["redis"] = {"status": "not_initialized"}

    # 5. RabbitMQ Probe
    try:
        import aio_pika

        from src.config import settings

        # Quick connection attempt
        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL, timeout=2)
        await connection.close()
        health["probes"]["rabbitmq"] = {"status": "connected"}
    except Exception as e:
        health["probes"]["rabbitmq"] = {"status": "error", "message": str(e)}
        health["status"] = "degraded"

    return DataResponseStruct(data=health)


@router.get(
    "/status",
    response_model=DataResponse[dict[str, Any]],
    dependencies=[Depends(require_tier(["admin", "enterprise"]))],
)
async def get_system_status():
    """Returns the status of various system components and circuit breakers."""
    return DataResponseStruct(
        data={
            "status": "operational",
            "circuits": {
                "pricing": {
                    "state": pricing_circuit.state.value,
                    "failure_count": pricing_circuit.failure_count,
                },
                "database": {
                    "state": db_circuit.state.value,
                    "failure_count": db_circuit.failure_count,
                },
            },
        }
    )


@router.get(
    "/diagnostics/db",
    response_model=DataResponse[dict[str, Any]],
    dependencies=[Depends(require_tier(["enterprise", "admin"]))],
)
async def get_db_diagnostics(db: AsyncSession = Depends(get_async_db)):
    """
    High-Performance Database Diagnostics.
    Requires Enterprise tier for high-fidelity performance metrics.
    """
    return DataResponseStruct(data=await crud.get_system_health_dashboard(db))


@router.get(
    "/diagnostics/io",
    response_model=DataResponse[dict[str, Any]],
    dependencies=[Depends(require_tier(["enterprise", "admin"]))],
)
async def get_io_diagnostics(db: AsyncSession = Depends(get_async_db)):
    """
    PostgreSQL 16 I/O Performance Audit.
    Requires Enterprise tier.
    """
    return DataResponseStruct(data=await crud.get_io_performance_audit(db))
@router.get(
    "/signals",
    response_model=DataResponse[list[dict[str, Any]]],
    dependencies=[Depends(require_tier(["free", "pro", "enterprise", "admin"]))],
)
async def get_signals(limit: int = 10, db: AsyncSession = Depends(get_async_db)):
    """
    Unified signal feed for the dashboard (Telemetry).
    """
    signals = await crud.get_recent_signals(db, limit)
    return DataResponseStruct(data=signals)
    return DataResponseStruct(data=signals)
