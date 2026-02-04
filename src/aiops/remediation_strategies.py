import structlog
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

logger = structlog.get_logger()

class RemediationStrategy(ABC):
    """Abstract base class for all AIOps remediation strategies."""
    @abstractmethod
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
        pass

class RestartServiceStrategy(RemediationStrategy):
    """Strategy to restart a specific Docker service."""
    supported_types = ["high_latency", "error_spike"]
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
# ...
class RetrainModelStrategy(RemediationStrategy):
    """Strategy to trigger ML pipeline retraining."""
    supported_types = ["data_drift", "performance_degradation"]
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
# ...
class PurgeCacheStrategy(RemediationStrategy):
    """Strategy to purge Redis cache."""
    supported_types = ["stale_data", "cache_inconsistency"]
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):


class RemediationRegistry:
    """Registry to map anomaly types to remediation strategies."""
    def __init__(self):
        self._strategies: Dict[str, RemediationStrategy] = {}

    def register(self, anomaly_type: str, strategy: RemediationStrategy):
        self._strategies[anomaly_type] = strategy

    def get_strategy(self, anomaly_type: str) -> List[RemediationStrategy]:
        """Returns a list of strategies for a given anomaly type."""
        strategy = self._strategies.get(anomaly_type)
        return [strategy] if strategy else []
