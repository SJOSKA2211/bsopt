from typing import Any

import structlog

from src.ml.autonomous_pipeline import AutonomousMLPipeline

logger = structlog.get_logger()

import concurrent.futures
from typing import Any

import structlog

from src.ml.autonomous_pipeline import AutonomousMLPipeline

logger = structlog.get_logger()

class MLPipelineTrigger:
    """
    Triggers ML retraining pipelines in the background.
    """
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def trigger_retraining(self) -> bool:
        """🚀 NON-BLOCKING: Triggers retraining in a background thread."""
        logger.info("ml_pipeline_trigger_async", status="starting", config=self.config)
        
        def _run_pipeline():
            try:
                pipeline = AutonomousMLPipeline(self.config)
                pipeline.run()
                logger.info("ml_pipeline_background_complete")
            except Exception as e:
                logger.error("ml_pipeline_background_failed", error=str(e))

        # Submit to executor and return immediately
        self.executor.submit(_run_pipeline)
        return True
