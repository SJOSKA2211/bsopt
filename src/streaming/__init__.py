import sys
from typing import TYPE_CHECKING, List
from src.utils.lazy_import import lazy_import, preload_modules

# ============================================================================
# PUBLIC API
# ============================================================================
__all__ = [
    # Kafka
    "MarketDataProducer", "MarketDataConsumer",
]

# These imports are only for type checking and will not be executed at runtime
if TYPE_CHECKING:
    from .kafka_producer import MarketDataProducer
    from .kafka_consumer import MarketDataConsumer

_import_map = {
    "MarketDataProducer": ".kafka_producer",
    "MarketDataConsumer": ".kafka_consumer",
}

def __getattr__(name: str):
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])

def __dir__() -> List[str]:
    return sorted(__all__)

def preload_streaming_modules():
    """Preload critical streaming modules for faster startup in production."""
    preload_modules(__name__, _import_map, {"MarketDataProducer", "MarketDataConsumer"})
