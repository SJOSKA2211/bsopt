"""
SHM Registry and Initialization Logic.
Centralizes all shared memory segments used in BS-OPT.
"""

from multiprocessing import shared_memory

import structlog

# Constants from shm_mesh (re-defined or imported to avoid circularity if needed)
# For now, we define them here or import them carefully.
from core.shared.shm_mesh import (
    BUFFER_CAPACITY,
    EXEC_BUFFER_CAPACITY,
    EXEC_SIZE,
    GREEKS_BUFFER_CAPACITY,
    GREEKS_MAP_SIZE,
    GREEKS_SIZE,
    ORDER_BUFFER_CAPACITY,
    ORDER_SIZE,
    RISK_STATE_DTYPE,
    SHM_EXEC_NAME,
    SHM_GREEKS_NAME,
    SHM_NAME,
    SHM_ORDER_NAME,
    SHM_RISK_NAME,
    TICK_SIZE,
)

logger = structlog.get_logger(__name__)

SHM_CONFIGS = [
    {
        "name": "market_mesh",
        "size": 50 * 1024 * 1024,
        "description": "General market data dictionary (msgspec)",
    },
    {
        "name": SHM_NAME,
        "size": (TICK_SIZE * BUFFER_CAPACITY) + 8,
        "description": "Lock-free Market Tick Ring Buffer",
    },
    {
        "name": SHM_ORDER_NAME,
        "size": (ORDER_SIZE * ORDER_BUFFER_CAPACITY) + 8,
        "description": "Order Command Buffer",
    },
    {
        "name": SHM_EXEC_NAME,
        "size": (EXEC_SIZE * EXEC_BUFFER_CAPACITY) + 8,
        "description": "Execution Status Buffer",
    },
    {"name": SHM_RISK_NAME, "size": RISK_STATE_DTYPE.itemsize, "description": "Risk State Buffer"},
    {
        "name": SHM_GREEKS_NAME,
        "size": (GREEKS_SIZE * GREEKS_BUFFER_CAPACITY) + 8,
        "description": "Greeks Stream Buffer",
    },
    {
        "name": "greeks_snapshot",
        "size": GREEKS_MAP_SIZE,
        "description": "Greeks Snapshot Map",
    },
]


def initialize_all_shm(force: bool = False):
    """
    Pre-allocates all SHM segments.
    If force=True, unlinks existing segments first.
    """
    for config in SHM_CONFIGS:
        name = config["name"]
        size = config["size"]

        if force:
            try:
                existing = shared_memory.SharedMemory(name=name)
                existing.close()
                existing.unlink()
                logger.info("shm_force_unlinked", name=name)
            except FileNotFoundError:
                pass

        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            # Initialize with zeros
            shm.buf[:size] = b"\x00" * size
            shm.close()
            logger.info("shm_initialized", name=name, size=size)
        except FileExistsError:
            # Check size
            shm = shared_memory.SharedMemory(name=name)
            if shm.size != size:
                logger.warning(
                    "shm_size_mismatch_recreating", name=name, existing=shm.size, expected=size
                )
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                shm.buf[:size] = b"\x00" * size
                shm.close()
            else:
                shm.close()
                logger.info("shm_already_exists", name=name)
        except Exception as e:
            logger.error("shm_init_failed", name=name, error=str(e))


if __name__ == "__main__":
    initialize_all_shm(force=True)
