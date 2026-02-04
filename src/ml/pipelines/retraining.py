import asyncio
import structlog
from typing import Dict, Any, Optional
from src.ml.pipelines.orchestrator import MLOrchestrator
from src.aiops.data_drift_detector import DataDriftDetector

logger = structlog.get_logger()

class NeuralGreeksRetrainer:
    """
    Automated Retrainer for SOTA Neural Greeks Engine.
    Features automated data drift detection and distributional validation.
    """
    def __init__(self, n_samples: int = 10000):
        self.orchestrator = MLOrchestrator()
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

    async def retrain_now(self, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Triggers a full retraining pipeline with pre-validation.
        """
        logger.info("triggering_neural_greeks_retraining", samples=self.n_samples)
        
        # 🚀 SECURITY: Stop retraining if data is corrupted or shifted
        if data is not None and not await self._validate_data(data):
            raise ValueError("Retraining aborted due to significant data drift.")
        
        try:
            result = await self.orchestrator.run_training_pipeline(
                model_type="nn",
                promote_to_production=True,
                n_samples=self.n_samples
            )
            logger.info("retraining_completed", run_id=result.get("run_id"), r2=result.get("r2"))
            return result
        except Exception as e:
            logger.error("retraining_failed", error=str(e))
            raise e
