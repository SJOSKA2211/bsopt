import os

import structlog

logger = structlog.get_logger()

def set_thread_affinity(core_id: int) -> None:
    """
    Sets the CPU affinity for the current thread/process to the specified core.
    
    Args:
        core_id (int): The CPU core ID to pin execution to.
    """
    try:
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {core_id})
            logger.info("thread_affinity_set", core=core_id)
        else:
            logger.debug("thread_affinity_not_supported", platform=os.name)
    except Exception as e:
        logger.warning("thread_affinity_failed", core=core_id, error=str(e))
