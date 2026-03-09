import logging
import os

import anyio
import torch
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponse
from src.database import crud, get_async_db
from src.security.auth import require_tier
from src.shared.shm_mesh import SharedMemoryRingBuffer
from src.utils.circuit_breaker import db_circuit, pricing_circuit

router = APIRouter(prefix="/system", tags=["System"], default_response_class=MsgspecJSONResponse)
logger = logging.getLogger(__name__)


# Global Probe Cache
_shm_probe = None


@router.get("/health/deep")
async def get_deep_health() -> DataResponse:
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

    # 2. Lazy CUDA Probe
    if os.getenv("BSOPT_USE_GPU", "0") == "1":
        try:
            cuda_available = torch.cuda.is_available()
            health["probes"]["cuda"] = {
                "status": "available" if cuda_available else "missing",
                "device": torch.cuda.get_device_name(0) if cuda_available else None,
            }
        except Exception:
            health["probes"]["cuda"] = {"status": "error"}
    else:
        health["probes"]["cuda"] = {"status": "disabled_by_config"}

    # 3. WASM OPA Probe
    wasm_path = "policies/authz.wasm"
    try:
        exists = await anyio.to_thread.run_sync(os.path.exists, wasm_path)
    except Exception:
        exists = False

    health["probes"]["wasm_opa"] = {
        "status": "verified" if exists else "missing",
        "path": wasm_path,
    }

    return DataResponse(data=health)


@router.get("/status")
async def get_system_status() -> DataResponse:
    """Returns the status of various system components and circuit breakers."""
    return DataResponse(
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


@router.get("/diagnostics/db", dependencies=[Depends(require_tier("enterprise"))])
async def get_db_diagnostics(db: AsyncSession = Depends(get_async_db)) -> DataResponse:
    """
    High-Performance Database Diagnostics.
    Requires Enterprise tier for high-fidelity performance metrics.
    """
    return DataResponse(data=await crud.get_system_health_dashboard(db))


@router.get("/diagnostics/io", dependencies=[Depends(require_tier("enterprise"))])
async def get_io_diagnostics(db: AsyncSession = Depends(get_async_db)) -> DataResponse:
    """
    PostgreSQL 16 I/O Performance Audit.
    Requires Enterprise tier.
    """
    return DataResponse(data=await crud.get_io_performance_audit(db))
