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
        """Triggers retraining pipeline asynchronously via Docker."""
        logger.info("ml_pipeline_trigger", status="attempting_retraining", config=self.config)

        try:
            import os
            import subprocess

            compose_bin = os.path.join(os.getcwd(), "docker-compose")
            # OPTIMIZED: Run asynchronously using the central MLOps worker
            subprocess.Popen(
                [
                    compose_bin,
                    "exec",
                    "-d",
                    "mlops-worker",
                    "mlflow",
                    "run",
                    ".",
                    "-e",
                    "train_regressor",
                    "-P",
                    f"ticker={self.config.get('ticker', 'AAPL')}",
                    "--experiment-name",
                    f"manual_trigger_{self.config.get('ticker', 'AAPL')}",
                    "--env-manager",
                    "local",
                ]
            )
            logger.info(
                "ml_pipeline_trigger",
                status="success",
                message="ML retraining job dispatched to mlops-worker.",
            )
            return True
        except Exception as e:
            logger.error(
                "ml_pipeline_trigger",
                status="failure",
                error=str(e),
                message="Failed to dispatch ML retraining job.",
            )
            return False
