from typing import Any

import structlog

from src.aiops.data_drift_detector import DataDriftDetector
from src.ml.training.train_v2 import train_neural_network

logger = structlog.get_logger()

class NeuralGreeksRetrainer:
    """
    Automated Retrainer for OPTIMIZED Neural Greeks Engine (V2).
    """
    def __init__(self, n_samples: int = 10000):
        self.drift_detector = DataDriftDetector()
        self.n_samples = n_samples

    async def _validate_data(self, data: Any) -> bool:
        """
        🚀 OPTIMIZATION: Perform distributional shift analysis before retraining.
        """
        drift_report = self.drift_detector.detect_drift(data)
        if drift_report.get("is_drift_detected", False):
            logger.error("retraining_aborted_data_drift", drift_report=drift_report)
            return False
        return True

    async def retrain_now(self, data: Any | None = None) -> dict[str, Any]:
        """
        Triggers a full retraining pipeline with pre-validation.
        """
        logger.info("triggering_neural_greeks_retraining_v2", samples=self.n_samples)
        
        if data is not None and not await self._validate_data(data):
            raise ValueError("Retraining aborted due to significant data drift.")
        
        try:
            # Call synchronous training logic in a thread or directly if fast enough (blocking for now as it's compute intensive)
            # ideally run in process pool
            logger.info("starting_training_loop")
            best_model_path = train_neural_network(n_samples=self.n_samples)
            
            logger.info("retraining_completed", model_path=str(best_model_path))
            return {"status": "success", "model_path": str(best_model_path)}
        except Exception as e:
            logger.error("retraining_failed", error=str(e))
            raise e
