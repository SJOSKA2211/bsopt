from fastapi import APIRouter

from src.utils.circuit_breaker import db_circuit, pricing_circuit

router = APIRouter(prefix="/system", tags=["System"])

import os

import torch

from src.shared.shm_mesh import SharedMemoryRingBuffer


@router.get("/health/deep")
async def get_deep_health():
    """🚀 SINGULARITY: High-fidelity stack probe."""
    health = {"status": "operational", "probes": {}}
    
    # 1. SHM Mesh Probe
    try:
        shm = SharedMemoryRingBuffer(create=False)
        head = shm.get_head()
        health["probes"]["shm_mesh"] = {"status": "connected", "head": head}
    except Exception as e:
        health["probes"]["shm_mesh"] = {"status": "corrupted", "error": str(e)}
        health["status"] = "degraded"

    # 2. CUDA GPU Probe
    try:
        cuda_available = torch.cuda.is_available()
        health["probes"]["cuda"] = {
            "status": "available" if cuda_available else "missing",
            "device": torch.cuda.get_device_name(0) if cuda_available else None
        }
    except Exception:
        health["probes"]["cuda"] = {"status": "error"}

    # 3. WASM OPA Probe
    wasm_path = "policies/authz.wasm"
    health["probes"]["wasm_opa"] = {
        "status": "verified" if os.path.exists(wasm_path) else "missing",
        "path": wasm_path
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
                "failure_count": pricing_circuit.failure_count
            },
            "database": {
                "state": db_circuit.state.value,
                "failure_count": db_circuit.failure_count
            }
        }
    }
