import os
import sys
from typing import TYPE_CHECKING, Any

from src.utils.lazy_import import get_import_stats, lazy_import, preload_modules

# PUBLIC API
__all__ = [
    # Forecasting
    "PriceTFTModel",
    # Reinforcement Learning
    "TradingEnvironment",
    "AugmentedRLAgent",
    # Federated Learning
    "FederatedLearningCoordinator",
    # Data Processing
    "DataNormalizer",
]

# TYPE HINTS (Static Analysis Only - Zero Runtime Cost)
if TYPE_CHECKING:
    from .data_loader import DataNormalizer
    from .federated_learning.coordinator import FederatedLearningCoordinator
    from .forecasting.tft_model import PriceTFTModel
    from .reinforcement_learning.augmented_agent import AugmentedRLAgent
    from .reinforcement_learning.trading_env import TradingEnvironment

# LAZY IMPORT MAPPING
_import_map = {
    # Forecasting (PyTorch-based)
    "PriceTFTModel": ".forecasting.tft_model",
    # RL (Ray + Torch)
    "TradingEnvironment": ".reinforcement_learning.trading_env",
    "AugmentedRLAgent": ".reinforcement_learning.augmented_agent",
    # Federated Learning (PySyft + Torch)
    "FederatedLearningCoordinator": ".federated_learning.coordinator",
    # Data (Lightweight - can preload)
    "DataNormalizer": ".data_loader",
}


# RUNTIME LAZY LOADING
def __getattr__(name: str) -> Any:
    """
    PEP 562: Module-level __getattr__ for lazy imports.
    This is called when an attribute is accessed that doesn't exist yet.
    We import it on-demand and cache it in the module.
    """
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])


def __dir__() -> list[str]:
    """
    PEP 562: Define what dir(module) returns.
    This ensures tab-completion and introspection work correctly.
    """
    return sorted(__all__)


# PRODUCTION PRELOADING
def preload_critical_modules() -> None:
    """
    Preload lightweight modules in production for faster first request.
    Call this from the application startup hook (e.g., FastAPI @app.on_event("startup")).
    """
    critical_modules = {
        "DataNormalizer",
    }
    preload_modules(__name__, _import_map, critical_modules)


# Auto-preload in production environments
if os.getenv("ENVIRONMENT") == "production" and os.getenv("PRELOAD_ML_MODULES") == "true":
    preload_critical_modules()


# DIAGNOSTICS
def get_ml_import_stats() -> dict[str, Any]:
    """Get import statistics for ML module."""
    return get_import_stats()
