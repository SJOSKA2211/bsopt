import asyncio
import structlog
from typing import Dict, Any
from src.ml.pipelines.orchestrator import MLOrchestrator

logger = structlog.get_logger()

class NeuralGreeksRetrainer:
    """
    Automated Retrainer for SOTA Neural Greeks Engine.
    Designed to be triggered by Airflow/Prefect DAGs.
    """
    def __init__(self, n_samples: int = 10000):
        self.orchestrator = MLOrchestrator()
        self.n_samples = n_samples

    async def retrain_now(self) -> Dict[str, Any]:
        """
        Triggers a full retraining and evaluation pipeline for Neural Greeks.
        """
        logger.info("triggering_neural_greeks_retraining", samples=self.n_samples)
        
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
