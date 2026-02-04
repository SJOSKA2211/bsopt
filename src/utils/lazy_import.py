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
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
import structlog
from importlib import import_module
from contextlib import contextmanager
import structlog

logger = structlog.get_logger(__name__)

# Global state
_import_locks: Dict[str, threading.RLock] = {}
_import_times: Dict[str, float] = {}
_import_stack: threading.local = threading.local()
_failed_imports: Dict[str, Exception] = {}

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
                module = import_module(module_path, package=package_name)
                attr = getattr(module, attr_name)
                
                # Cache in the parent module
                setattr(cache_module, attr_name, attr)
                
                # Record timing
                elapsed = time.perf_counter() - start_time
                _import_times[cache_key] = elapsed
                
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
    cache_module_override: Any = None,
    service_type: Optional[str] = None
):
    """
    Eagerly load a set of modules with service-awareness.
    Useful for 'warming up' critical paths in production without over-loading.
    """
    current_service = service_type or os.environ.get("SERVICE_TYPE", "api")
    
    logger.info(
        "preload_start",
        package=package_name,
        service=current_service
    )

    start_time = time.perf_counter()
    cache_module = cache_module_override or sys.modules[package_name]

    for attr_name in attributes:
        # Optimization: Skip preloading heavy ML modules for simple API instances
        if current_service == "api" and "ml" in package_name and "core" not in attr_name:
            continue
            
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
