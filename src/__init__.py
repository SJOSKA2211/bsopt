# BSOpt Optimized Core
"""
Advanced Financial Manifold for Transdimensional Derivative Pricing.
"""

__version__ = "2.5.0"

import sys
from typing import Any

from .utils.lazy_import import lazy_import

_import_map = {
    "aiops": ".aiops",
    "api": ".api",
    "audit": ".audit",
    "auth": ".auth",
    "config": ".config",
    "database": ".database",
    "ml": ".ml",
    "pricing": ".pricing",
    "services": ".services",
    "shared": ".shared",
    "streaming": ".streaming",
    "utils": ".utils",
    "workers": ".workers",
}


def __getattr__(name: str) -> Any:
    if name in _import_map:
        return lazy_import(__name__, _import_map, name, sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(_import_map.keys()) + ["__version__"])
