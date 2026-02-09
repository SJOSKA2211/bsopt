# BSOpt Optimized Core
"""
God-Mode Financial Manifold for Transdimensional Derivative Pricing.
"""

__version__ = "2.5.0"

import sys

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

def __getattr__(name: str):
    if name in _import_map:
        return lazy_import(__name__, _import_map, name, sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(_import_map.keys()) + ["__version__"])
