import sys
import os
from typing import TYPE_CHECKING, List
from src.utils.lazy_import import lazy_import, preload_modules, get_import_stats

# ============================================================================
# PUBLIC API
# ============================================================================
__all__ = [
    # Forecasting
    "PriceTFTModel", "TFTModel",
    # Reinforcement Learning
    "TradingEnvironment", "AugmentedRLAgent",
    # Federated Learning
    "FederatedLearningCoordinator",
    # Data Processing
    "DataNormalizer",
]

# ============================================================================
# TYPE HINTS (Static Analysis Only - Zero Runtime Cost)
# ============================================================================
if TYPE_CHECKING:
    from .forecasting.tft_model import PriceTFTModel, TFTModel
    from .reinforcement_learning.trading_env import TradingEnvironment
    from .rl.augmented_agent import AugmentedRLAgent
    from .federated_learning.coordinator import FederatedLearningCoordinator
    from .data_loader import DataNormalizer

# ============================================================================
# LAZY IMPORT MAPPING
# ============================================================================
_import_map = {
    # Forecasting (PyTorch-based)
    "PriceTFTModel": ".forecasting.tft_model",
    "TFTModel": ".forecasting.tft_model",
    
    # RL (Ray + Torch)
    "TradingEnvironment": ".reinforcement_learning.trading_env",
    "AugmentedRLAgent": ".rl.augmented_agent",
    
    # Federated Learning (PySyft + Torch)
    "FederatedLearningCoordinator": ".federated_learning.coordinator",
    
    # Data (Lightweight - can preload)
    "DataNormalizer": ".data_loader",
}

# ============================================================================
# RUNTIME LAZY LOADING
# ============================================================================
def __getattr__(name: str):
    """
    PEP 562: Module-level __getattr__ for lazy imports.
    This is called when an attribute is accessed that doesn't exist yet.
    We import it on-demand and cache it in the module.
    """
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])

def __dir__() -> List[str]:
    """
    PEP 562: Define what dir(module) returns.
    This ensures tab-completion and introspection work correctly.
    """
    return sorted(__all__)

# ============================================================================
# PRODUCTION PRELOADING
# ============================================================================
def preload_critical_modules():
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

# ============================================================================
# DIAGNOSTICS
# ============================================================================
def get_ml_import_stats():
    """Get import statistics for ML module."""
    return get_import_stats()
