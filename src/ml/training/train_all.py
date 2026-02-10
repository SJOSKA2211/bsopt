"""
Master Training Pipeline Entry Point
====================================

Orchestrates the Autonomous ML Pipeline.
Now uses the Advanced `AutonomousMLPipeline` class.
"""

import asyncio
import os

import structlog

from src.config import get_settings
from src.ml.autonomous_pipeline import AutonomousMLPipeline

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
        "use_warm_start": True
    }
    
    logger.info("initializing_autonomous_pipeline", config=config)
    pipeline = AutonomousMLPipeline(config)
    
    try:
        study = await pipeline.run()
        if study:
            logger.info("pipeline_success", best_value=study.best_value)
        else:
            logger.info("pipeline_skipped_no_drift")
    except Exception as e:
        logger.critical("pipeline_fatal_error", error=str(e))
        raise

if __name__ == "__main__":
    asyncio.run(train_all())
