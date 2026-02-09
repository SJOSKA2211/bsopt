import os
import structlog

logger = structlog.get_logger(__name__)

def set_thread_affinity(core_id: int):
    """
    Pins the current thread/process to a specific CPU core.
    Fails gracefully if OS does not support affinity setting.
    """
    try:
        os.sched_setaffinity(0, {core_id})
        logger.info("thread_pinned", core=core_id)
    except AttributeError:
        logger.warning("affinity_not_supported_on_platform")
    except Exception as e:
        logger.error("affinity_setting_failed", error=str(e), core=core_id)
