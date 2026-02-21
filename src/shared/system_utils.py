import gc
import os
import sys

import structlog

logger = structlog.get_logger(__name__)

def set_thread_affinity(core_id: int):
    """
    Pins the current thread/process to a specific CPU core.
    OPTIMIZED: Checks core availability before pinning.
    """
    try:
        if hasattr(os, "sched_setaffinity"):
            # Check actual available cores to prevent index error
            available_cores = os.sched_getaffinity(0)
            if core_id in available_cores:
                os.sched_setaffinity(0, {core_id})
                logger.info("thread_pinned", core=core_id)
            else:
                logger.warning("core_not_available_for_affinity", requested=core_id, available=available_cores)
        else:
            logger.debug("affinity_not_supported_on_platform")
    except Exception as e:
        logger.error("affinity_setting_failed", error=str(e), core=core_id)

def tune_gc(mode: str = "high_frequency"):
    """
    OPTIMIZED: Tune Python GC for specific workloads.
    'high_frequency' mode reduces stop-the-world latency by increasing thresholds.
    """
    if mode == "high_frequency":
        # Increase generation thresholds to delay collections during bursts
        gc.set_threshold(50000, 10, 10)
        logger.info("gc_tuned_high_frequency")
    elif mode == "batch":
        gc.set_threshold(700, 10, 10) # Standard
    
    # Enable aggressive collection tracking in dev
    if os.getenv("ENVIRONMENT") == "dev":
        gc.set_debug(gc.DEBUG_STATS)

def set_process_priority(priority: int = -10):
    """
    Sets the OS-level process priority (niceness).
    -10 is higher priority, 0 is standard.
    """
    if sys.platform != 'win32':
        try:
            os.nice(priority)
            logger.info("process_priority_escalated", niceness=priority)
        except PermissionError:
            logger.warning("priority_escalation_denied_insufficient_privileges")

def get_memory_usage_mb() -> float:
    """Returns the RSS memory usage of the current process in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)
