import os

import anyio
import torch
from fastapi import APIRouter

from src.shared.shm_mesh import SharedMemoryRingBuffer
from src.utils.circuit_breaker import db_circuit, pricing_circuit

router = APIRouter(prefix="/system", tags=["System"])


# Global Probe Cache
_shm_probe = None

@router.get("/health/deep")
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
        health["probes"]["shm_mesh"] = {"status": "corrupted", "error": str(e)}
        health["status"] = "degraded"
        _shm_probe = None # Reset on failure

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
    exists = await anyio.to_thread.run_sync(os.path.exists, wasm_path)
    health["probes"]["wasm_opa"] = {
        "status": "verified" if exists else "missing",
        "path": wasm_path,
    }

    return health


@router.get("/status")
async def get_system_status():
    """Returns the status of various system components and circuit breakers."""
    return {
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
