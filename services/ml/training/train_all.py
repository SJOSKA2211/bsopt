"""
Master Training Pipeline Entry Point

Orchestrates the Autonomous ML Pipeline.
Now uses the Advanced `MLPipeline` class.
"""

import asyncio
import os

import structlog

from services.config import get_settings
from services.ml.pipeline import MLPipeline

logger = structlog.get_logger()


async def train_all():
    """
    Execute the unified autonomous pipeline.
    """
    settings = get_settings()

    # Configuration for the pipeline
    config = {
        "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", "DEMO_KEY"),
        "provider": "auto",  # auto-select provider
        "db_url": settings.DATABASE_URL,
        "ticker": os.getenv("TICKER", "AAPL"),
        "study_name": os.getenv("STUDY_NAME", "autonomous_opt_v1"),
        "n_trials": int(os.getenv("N_TRIALS", "20")),
        "framework": os.getenv("FRAMEWORK", "xgboost"),
        "promotion_threshold": 0.85,
        "use_warm_start": True,
    }

    logger.info("initializing_autonomous_pipeline", config=config)
    pipeline = MLPipeline(config)

    try:
        model = await pipeline.run()
        if model:
            logger.info("pipeline_success", model_promoted=True, framework=config["framework"])
        else:
            logger.info("pipeline_skipped_no_drift")

    except Exception as e:
        logger.critical("pipeline_fatal_error", error=str(e))
        raise
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(train_all())
