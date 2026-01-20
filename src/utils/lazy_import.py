"""
Lazy import utilities with production-grade features.
Features:
- Thread-safe lazy loading
- Import performance monitoring
- Circular import detection
- Graceful error handling
- Module preloading for production
"""
import sys
import time
import threading
import importlib
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
import structlog
from contextlib import contextmanager
import structlog

logger = structlog.get_logger(__name__)

# Global state
_import_locks: Dict[str, threading.RLock] = {}
_import_times: Dict[str, float] = {}
_import_stack: threading.local = threading.local()
_failed_imports: Dict[str, Exception] = {}

# Allowlist for modules that can be lazy-imported.
# This prevents arbitrary module loading if an attacker can control module_path or package_name.
_module_allowlist = {
    "src.ml": [
        "src.ml.rl.augmented_agent",
        "src.ml.training.train",
        "src.ml.pipelines.orchestrator",
        "src.ml.evaluation.compare_models",
        "src.ml.data_loader",
        "src.ml.drift",
        "src.ml.scraper",
        "src.ml.serving.serve_model",
        "src.ml.serving.serve",
        "src.ml.serving.quantization",
        "src.ml.reinforcement_learning.train",
        "src.ml.reinforcement_learning.online_agent",
        "src.ml.federated_learning.coordinator",
        "src.ml.pipelines.retraining",
        "src.ml.pipelines.sentiment_ingest",
        "src.ml.utils.inference",
        "src.ml.utils.optimization",
        "src.ml.utils.rollback",
        "src.ml.utils.versioning",
        "src.ml.utils.distributed",
    ],
    "src.pricing": [
        "src.pricing.black_scholes",
        "src.pricing.factory",
        "src.pricing.implied_vol",
        "src.pricing.exotic",
        "src.pricing.models.heston_fft",
        "src.pricing.quant_utils",
        "src.pricing.vectorized_black_scholes",
        "src.pricing.quantum_backend",
        "src.pricing.quantum_pricing",
    ],
    "src.data": [
        "src.data.pipeline",
        "src.data.router",
    ],
    "src.utils": [
        "src.utils.cache",
        "src.utils.circuit_breaker",
        "src.utils.lazy_import",
        "src.utils.sanitization",
        "src.utils.errors",
        "src.utils.dashboard",
    ],
    "src.tasks": [
        "src.tasks.pricing_tasks",
        "src.tasks.ml_tasks",
        "src.tasks.trading_tasks",
        "src.tasks.data_tasks",
        "src.tasks.email_tasks",
        "src.tasks.audit_tasks",
        "src.tasks.celery_app",
        "src.tasks.graceful_shutdown",
    ],
    "src.security": [
        "src.security.auth",
        "src.security.password",
        "src.security.audit",
        "src.security.breach_notification",
        "src.security.rate_limit",
        "src.security.opa",
        "src.security.mtls",
    ],
    "src.database": [
        "src.database.models",
        "src.database.crud",
        "src.database.verify",
    ],
    "src.streaming": [
        "src.streaming.mesh_producer",
        "src.streaming.health",
        "src.streaming.kafka_consumer",
        "src.streaming.kafka_producer",
        "src.streaming.consumer",
        "src.streaming.producer",
        "src.streaming.analytics",
    ],
    "src.api": [
        "src.api.routes.auth",
        "src.api.routes.pricing",
        "src.api.routes.users",
        "src.api.routes.ml",
        "src.api.routes.websocket",
        "src.api.routes.system",
        "src.api.routes.debug",
        "src.api.middleware.security",
        "src.api.middleware.logging",
        "src.api.middleware.idempotency",
        "src.api.middleware.profiling",
        "src.api.graphql.schema",
        "src.api.schemas.auth",
        "src.api.schemas.common",
        "src.api.schemas.pricing",
        "src.api.schemas.user",
        "src.api.schemas.ml",
        "src.api.schemas.streaming",
        "src.api.schemas.system",
        "src.api.exceptions",
    ],
    "src.cli": [
        "src.cli.portfolio",
        "src.cli.main",
        "src.cli.auth",
        "src.cli.config",
        "src.cli.data",
        "src.cli.pricing",
        "src.cli.ml",
        "src.cli.reporting",
    ],
    "src.aiops": [
        "src.aiops.aiops_orchestrator",
        "src.aiops.autoencoder_detector",
        "src.aiops.data_drift_detector",
        "src.aiops.docker_remediator",
        "src.aiops.ml_pipeline_trigger",
        "src.aiops.prometheus_adapter",
        "src.aiops.redis_remediator",
        "src.aiops.self_healing_orchestrator",
        "src.aiops.timeseries_anomaly_detector",
    ],
    "src.shared": [
        "src.shared.observability",
        "src.shared.security",
    ],
}


class LazyImportError(ImportError):
    """Raised when a lazy import fails with additional context."""
    pass

class CircularImportError(ImportError):
    """Raised when a circular import is detected."""
    pass

def _get_import_lock(module_name: str) -> threading.RLock:
    """Get or create a lock for a specific module import."""
    if module_name not in _import_locks:
        _import_locks[module_name] = threading.RLock()
    return _import_locks[module_name]

@contextmanager
def _track_import_stack(module_name: str):
    """Context manager to track and detect circular imports."""
    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    
    if module_name in _import_stack.modules:
        stack = ' -> '.join(_import_stack.modules)
        raise CircularImportError(
            f"Circular import detected: {stack} -> {module_name}"
        )
    
    _import_stack.modules.append(module_name)
    try:
        yield
    finally:
        _import_stack.modules.pop()

def lazy_import(
    package_name: str,
    import_map: Dict[str, str],
    attr_name: str,
    cache_module: Any
) -> Any:
    """
    Thread-safe lazy import with monitoring and error handling.
    """
    if attr_name not in import_map:
        available = ', '.join(sorted(import_map.keys()))
        raise AttributeError(
            f"module {package_name!r} has no attribute {attr_name!r}. "
            f"Available: {available}"
        )

    # Check if previously failed
    cache_key = f"{package_name}.{attr_name}"
    if cache_key in _failed_imports:
        original_error = _failed_imports[cache_key]
        raise LazyImportError(
            f"Previous import of {cache_key} failed: {original_error}"
        ) from original_error

    module_path = import_map[attr_name]
    # Handle relative imports
    if module_path.startswith('.'):
        full_module_path = f"{package_name}{module_path}"
    else:
        full_module_path = module_path

    # --- SECURITY: Module Allowlisting ---
    # Ensure that both the package and the specific module are in the allowlist
    if package_name not in _module_allowlist or full_module_path not in _module_allowlist[package_name]:
        logger.error(
            "lazy_import_denied",
            package=package_name,
            attribute=attr_name,
            module=full_module_path,
            reason="Module not in allowlist"
        )
        raise LazyImportError(
            f"Attempted to lazy import unauthorized module: {full_module_path}"
        )

    # Thread-safe import
    with _get_import_lock(full_module_path):
        # Double-check if already imported by another thread
        if attr_name in cache_module.__dict__:
            return cache_module.__dict__[attr_name]

        start_time = time.perf_counter()
        try:
            with _track_import_stack(full_module_path):
                logger.debug(
                    "lazy_import_start",
                    package=package_name,
                    attribute=attr_name,
                    module=module_path
                )
                
                # Perform the import
                module = importlib.import_module(module_path, package=package_name)
                attr = getattr(module, attr_name)
                
                # Cache in the parent module
                setattr(cache_module, attr_name, attr)
                
                # Record timing
                elapsed = time.perf_counter() - start_time
                _import_times[cache_key] = elapsed
                print(f"DEBUG: Updated _import_times for {cache_key}. Size: {len(_import_times)}")
                
                logger.info(
                    "lazy_import_success",
                    package=package_name,
                    attribute=attr_name,
                    duration=f"{elapsed*1000:.2f}ms"
                )
                return attr
        except CircularImportError:
            raise
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            # Cache the failure
            _failed_imports[cache_key] = e
            logger.error(
                "lazy_import_failed",
                package=package_name,
                attribute=attr_name,
                error=str(e),
                duration=f"{elapsed*1000:.2f}ms"
            )
            raise LazyImportError(
                f"Failed to import {attr_name} from {package_name}: {e}"
            ) from e

def get_import_stats() -> Dict[str, Any]:
    """Get statistics about lazy imports."""
    return {
        'successful_imports': len(_import_times),
        'failed_imports': len(_failed_imports),
        'total_import_time': sum(_import_times.values()),
        'slowest_imports': sorted(
            _import_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
        'failures': {k: str(v) for k, v in _failed_imports.items()}
    }

def reset_import_stats():
    """Reset import statistics."""
    _import_times.clear()
    _failed_imports.clear()

def preload_modules(
    package_name: str,
    import_map: Dict[str, str],
    attributes: Iterable[str],
    cache_module_override: Any = None
):
    """
    Eagerly load a set of modules.
    Useful for 'warming up' critical paths in production.
    """
    logger.info(
        "preload_start",
        package=package_name,
        count=len(list(attributes))
    )

    start_time = time.perf_counter()
    cache_module = cache_module_override or sys.modules[package_name]

    for attr_name in attributes:
        try:
            lazy_import(package_name, import_map, attr_name, cache_module)
        except Exception as e:
            logger.warning(
                "preload_failed",
                package=package_name,
                attribute=attr_name,
                error=str(e)
            )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "preload_complete",
        package=package_name,
        duration=f"{elapsed:.2f}s"
    )
