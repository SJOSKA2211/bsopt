from fastapi import APIRouter, Depends
from src.utils.circuit_breaker import pricing_circuit, db_circuit
from src.security.auth import require_admin

router = APIRouter(prefix="/system", tags=["System"], dependencies=[Depends(require_admin())])

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
