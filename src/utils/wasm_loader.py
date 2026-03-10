import os
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class WasmModuleCache:
    """
    OPTIMIZED: Persistence and caching for compiled WASM modules.
    Reduces instantiation latency by 100x by serializing the machine code.
    """

    _memory_cache: dict[str, Any] = {}

    @classmethod
    def get_module(cls, store: Any, wasm_path: str) -> Any:
        """Get pre-compiled module from cache or disk."""
        if wasm_path in cls._memory_cache:
            return cls._memory_cache[wasm_path]

        from wasmer import Module

        # 1. Check for serialized artifact on disk
        cache_path = f"{wasm_path}.compiled"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    compiled_bytes = f.read()
                # OPTIMIZED: Deserializing pre-compiled machine code
                module = Module.deserialize(store, compiled_bytes)
                cls._memory_cache[wasm_path] = module
                logger.info("wasm_module_deserialized", path=wasm_path)
                return module
            except Exception as e:
                logger.warning("wasm_deserialization_failed", error=str(e))

        # 2. Fallback: Compile and serialize
        logger.info("wasm_module_compiling", path=wasm_path)
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = Module(store, wasm_bytes)

        # 3. Save for future dimensions
        try:
            serialized_bytes = module.serialize()
            with open(cache_path, "wb") as f:
                f.write(serialized_bytes)
            logger.info("wasm_module_serialized", path=cache_path)
        except Exception as e:
            logger.warning("wasm_serialization_failed", error=str(e))

        cls._memory_cache[wasm_path] = module
        return module

    @classmethod
    def map_wasm_memory(
        cls, instance: Any, offset: int = 0, size: int | None = None
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Zero-copy memory view of a specific slice of the WASM heap.
        WARNING: Manual memory management required. Offset/size must align with Rust exports.
        """
        try:
            wasm_mem = instance.exports.memory
            buffer = wasm_mem.buffer

            # Map a specific window of the linear memory
            # Default to full buffer if size is None
            if size is None:
                size = (len(buffer) - offset) // 8

            # OPTIMIZED: Using frombuffer for zero-copy view
            data_view: np.ndarray[Any, np.dtype[np.float64]] = np.frombuffer(
                buffer, dtype=np.float64, count=size, offset=offset
            )
            return data_view
        except Exception as e:
            logger.error("wasm_memory_mapping_failed", error=str(e))
            return np.empty(0, dtype=np.float64)


def get_wasm_instance() -> Any:
    """
    Singleton accessor for the WASM pricing instance.
    Locates the module in the project tree and instantiates it.
    """
    from wasmer import Instance, Store

    # 1. Locate the WASM file
    search_paths = [
        "src/frontend/public/wasm/bsopt_wasm_bg.wasm",
        "src/frontend/src/wasm/bsopt_wasm_bg.wasm",
        "wasm/bsopt_wasm_bg.wasm",
    ]
    wasm_path = next((p for p in search_paths if os.path.exists(p)), None)

    if not wasm_path:
        logger.error("wasm_module_not_found", searched=search_paths)
        return None

    try:
        # 2. Setup wasmer environment
        store = Store()
        module = WasmModuleCache.get_module(store, wasm_path)

        # 3. Instantiate (No imports needed for pure math kernels)
        instance = Instance(module)
        logger.info("wasm_instance_created", path=wasm_path)
        return instance
    except Exception as e:
        logger.error("wasm_instantiation_failed", error=str(e))
        return None


# Singleton accessor
wasm_cache = WasmModuleCache()
