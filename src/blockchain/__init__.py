"""
Blockchain Integration - DeFi, Oracles, IPFS
Web3.py loads entire EVM bytecode on import (~150MB). Defer until needed.
"""

import sys
from typing import TYPE_CHECKING, List

from src.utils.lazy_import import lazy_import

__all__ = [
    # DeFi Protocols
    "DeFiOptionsProtocol",
]

if TYPE_CHECKING:
    from .defi_options import DeFiOptionsProtocol

_import_map = {
    "DeFiOptionsProtocol": ".defi_options",
}


def __getattr__(name: str):
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])


def __dir__() -> list[str]:
    return sorted(__all__)
