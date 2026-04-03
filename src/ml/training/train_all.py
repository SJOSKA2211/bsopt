"""
Master Training Pipeline Entry Point

Orchestrates the Autonomous ML Pipeline.
Now uses the Advanced `MLPipeline` class.
"""

import asyncio
import os

import structlog

from src.ml.pipeline import MLPipeline
from src.shared.config import get_settings

logger = structlog.get_logger()


async def train_all():
    """
    Execute the unified autonomous pipeline.
    """
    settings = get_settings()

    from src.ml.training.base import TrainingConfig

    # Configuration for the pipeline
    config = TrainingConfig(
        framework=os.getenv("FRAMEWORK", "xgboost"),
        metadata={
            "api_key": os.getenv("ALPHA_VANTAGE_API_KEY") or "",
            "provider": "auto",
            "ticker": os.getenv("TICKER", settings.DEFAULT_TICKER),
            "study_name": os.getenv("STUDY_NAME", "autonomous_opt_v1"),
        }
    )

    logger.info("initializing_autonomous_pipeline", config=config)
    pipeline = MLPipeline(config)

    try:
        model = await pipeline.run()
        if model:
            logger.info("pipeline_success", model_promoted=True, framework=config.framework)
        else:
            logger.info("pipeline_skipped_no_drift")

    except Exception as e:
        logger.critical("pipeline_fatal_error", error=str(e))
        raise
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(train_all())
