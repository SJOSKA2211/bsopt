from typing import Any

import structlog

from src.ml.autonomous_pipeline import AutonomousMLPipeline

logger = structlog.get_logger()


class MLPipelineTrigger:
    """
    Triggers ML retraining pipelines.
    """

    def __init__(self, config: dict[str, Any]):
        if "ticker" not in config or "framework" not in config:
            raise ValueError("ML Pipeline config must contain 'ticker' and 'framework'")
        self.config = config

    def trigger_retraining(self) -> bool:
        """Triggers retraining pipeline."""
        logger.info("ml_pipeline_trigger", status="attempting_retraining", config=self.config)

<<<<<<< HEAD
        def _run_pipeline():
            try:
                pipeline = MLPipeline(self.config)
                import asyncio

                asyncio.run(pipeline.run())
                logger.info("ml_pipeline_background_complete")
            except Exception as e:
                logger.error("ml_pipeline_background_failed", error=str(e))

        # Submit to executor and return immediately
        self.executor.submit(_run_pipeline)
        return True
=======
        try:
            pipeline = AutonomousMLPipeline(self.config)
            pipeline.run()
            logger.info(
                "ml_pipeline_trigger",
                status="success",
                message="ML retraining pipeline triggered successfully.",
            )
            return True
        except Exception as e:
            logger.error(
                "ml_pipeline_trigger",
                status="failure",
                error=str(e),
                message="Failed to trigger ML retraining pipeline.",
            )
            return False
>>>>>>> 2a3ad5e0c (feat: Implement database optimization revamp, enhance ML and AIOps modules, and update frontend dependencies.)
