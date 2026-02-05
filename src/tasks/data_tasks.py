"""
Data Collection Tasks for Celery
=================================

Asynchronous data collection tasks for:
- Options data collection from yfinance
- Data pipeline execution
- Data validation and quality checks
- Scheduled data refresh
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from .celery_app import MLTask, celery_app

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Collection Tasks
# =============================================================================


@celery_app.task(
    bind=True,
    base=MLTask,
    queue="ml",
    priority=3,
    time_limit=1800,  # 30 minute limit
    soft_time_limit=1500,
)
def collect_options_data_task(
    self,
    symbols: list[str] | None = None,
    min_samples: int = 10000,
    max_samples: int = 50000,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Collect options data from market sources.

    Args:
        symbols: List of symbols to collect (None for defaults)
        min_samples: Minimum samples to collect
        max_samples: Maximum samples to collect
        validate: Apply data quality filters

    Returns:
        Collection report dict
    """
    logger.info("options_data_collection_start", symbols=symbols or "default symbols")

    try:
        from src.data.pipeline import DataPipeline, PipelineConfig, StorageBackend

        config = PipelineConfig(
            symbols=symbols
            or [
                "SPY",
                "QQQ",
                "IWM",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "TSLA",
                "AMD",
                "NFLX",
                "NIFTY",
                "BANKNIFTY",
            ],
            min_samples=min_samples,
            max_samples=max_samples,
            use_multi_source=True,
            validate_data=validate,
            storage_backend=StorageBackend.DATABASE,
            output_dir="data/training",
        )

        pipeline = DataPipeline(config)

        # Run the async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            report = loop.run_until_complete(pipeline.run())
        finally:
            loop.close()

        logger.info("data_collection_completed", samples_valid=report.get('samples_valid', 0))

        return {
            "task_id": self.request.id,
            "status": "success",
            "samples_collected": report.get("samples_collected", 0),
            "samples_valid": report.get("samples_valid", 0),
            "output_path": report.get("output_path", ""),
            "duration_seconds": report.get("duration_seconds", 0),
            "validation_rate": report.get("validation_rate", 0),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("data_collection_failed", error=str(e))
        raise


@celery_app.task(
    bind=True,
    base=MLTask,
    queue="ml",
    priority=4,
    time_limit=600,
    soft_time_limit=500,
)
def validate_collected_data_task(
    self,
    data_path: str,
) -> dict[str, Any]:
    """
    Validate previously collected data.

    Args:
        data_path: Path to the collected data directory

    Returns:
        Validation report dict
    """
    logger.info("data_validation_start", data_path=data_path)

    try:
        from pathlib import Path

        import pandas as pd

        data_dir = Path(data_path)

        # Load parquet data
        parquet_path = data_dir / "training_data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"No parquet file found at {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Basic validation
        validation_results: dict[str, Any] = {
            "total_samples": len(df),
            "n_features": len(df.columns) - 1,  # Exclude target
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        # Check target distribution
        target = df["target"] if "target" in df.columns else df.iloc[:, -1]
        validation_results["target_stats"] = {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }

        # Check for outliers (values beyond 5 std)
        outlier_threshold = 5
        outliers = ((target - target.mean()).abs() > outlier_threshold * target.std()).sum()
        validation_results["outliers"] = int(outliers)

        # Quality score
        quality_score = 1.0
        if validation_results["missing_values"] > 0:
            quality_score -= min(0.3, validation_results["missing_values"] / len(df))
        if validation_results["duplicate_rows"] > 0:
            quality_score -= min(0.2, validation_results["duplicate_rows"] / len(df))
        if validation_results["outliers"] > len(df) * 0.01:
            quality_score -= 0.1

        validation_results["quality_score"] = round(quality_score, 3)
        validation_results["passed"] = quality_score >= 0.7

        logger.info(
            "validation_complete", 
            quality_score=quality_score, 
            passed=validation_results['passed']
        )

        return {
            "task_id": self.request.id,
            "status": "success",
            "data_path": str(data_path),
            "validation": validation_results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("data_validation_failed", error=str(e))
        raise


@celery_app.task(
    bind=True,
    queue="ml",
    priority=2,
)
def check_data_freshness_task(self) -> dict[str, Any]:
    """
    Check if training data is fresh enough for use.

    Returns:
        Freshness check report
    """
    logger.info("data_freshness_check_start")

    try:
        import os
        from pathlib import Path

        data_dir = Path("data/training")

        if not data_dir.exists():
            return {
                "task_id": self.request.id,
                "status": "no_data",
                "message": "No training data directory found",
                "needs_refresh": True,
            }

        # Find latest data
        runs = sorted(data_dir.glob("pipeline_*"), reverse=True)

        if not runs:
            return {
                "task_id": self.request.id,
                "status": "no_data",
                "message": "No data runs found",
                "needs_refresh": True,
            }

        latest_run = runs[0]
        mtime = os.path.getmtime(latest_run)
        age_hours = (datetime.now().timestamp() - mtime) / 3600

        # Data older than 24 hours needs refresh
        max_age_hours = 24
        needs_refresh = age_hours > max_age_hours

        return {
            "task_id": self.request.id,
            "status": "success",
            "latest_run": str(latest_run.name),
            "age_hours": round(age_hours, 2),
            "max_age_hours": max_age_hours,
            "needs_refresh": needs_refresh,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("freshness_check_failed", error=str(e))
        return {
            "task_id": self.request.id,
            "status": "error",
            "error": str(e),
            "needs_refresh": True,
        }


@celery_app.task(
    bind=True,
    queue="batch",
    priority=1,
)
def refresh_materialized_views_task(self) -> dict[str, Any]:
    """
    Refreshes PostgreSQL materialized views for pre-aggregated statistics.
    """
    logger.info("refreshing_materialized_views_start")
    
    from sqlalchemy import text

    from src.shared.db import get_db_session
    
    db_session = get_db_session()
    try:
        # Refresh Market Stats
        db_session.execute(text("SELECT refresh_market_stats();"))
        
        # Refresh Portfolio Summary
        db_session.execute(text("SELECT refresh_portfolio_summary();"))
        
        # Refresh Trading Stats
        db_session.execute(text("SELECT refresh_trading_stats();"))

        # Refresh Model Drift Metrics
        db_session.execute(text("SELECT refresh_model_drift_metrics();"))
        
        db_session.commit()
        logger.info("materialized_views_refreshed_successfully")
        
        return {
            "status": "success",
            "views": [
                "market_stats_mv", 
                "portfolio_summary_mv", 
                "trading_stats_mv",
                "model_drift_metrics_mv"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("materialized_view_refresh_failed", error=str(e))
        db_session.rollback()
        raise self.retry(exc=e)
    finally:
        db_session.close()


# =============================================================================
# Periodic Data Collection
# =============================================================================


@celery_app.task(
    bind=True,
    queue="ml",
    priority=2,
)
def scheduled_data_collection(self) -> dict[str, Any]:
    """
    Scheduled task to collect data if needed.
    Called by Celery Beat.
    """
    logger.info("scheduled_data_collection_check")

    # Check if we need fresh data
    freshness = check_data_freshness_task.apply().get()

    if freshness.get("needs_refresh", True):
        logger.info("data_needs_refresh", reason=freshness.get("message", "Data too old"))

        # Trigger collection
        result = collect_options_data_task.apply_async()

        return {
            "task_id": self.request.id,
            "status": "collection_started",
            "collection_task_id": result.id,
            "reason": freshness.get("message", "Data too old"),
            "timestamp": datetime.now().isoformat(),
        }

    else:
        logger.info("data_is_fresh", age_hours=freshness.get('age_hours', 0))
        return {
            "task_id": self.request.id,
            "status": "skipped",
            "reason": f"Data is {freshness.get('age_hours', 0):.1f} hours old",
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Data Pipeline Chain
# =============================================================================


@celery_app.task(
    bind=True,
    queue="ml",
    priority=3,
)
def run_full_data_pipeline_task(
    self,
    symbols: list[str] | None = None,
    train_after_collection: bool = True,
) -> dict[str, Any]:
    """
    Run the full data pipeline: collect, validate, and optionally train.

    Args:
        symbols: Symbols to collect
        train_after_collection: Whether to trigger training after collection

    Returns:
        Pipeline execution report
    """
    logger.info("full_data_pipeline_start")

    try:
        # Step 1: Collect data
        collection_result = collect_options_data_task.apply(
            args=[symbols, 10000, 50000, True]
        ).get()

        if collection_result.get("status") != "success":
            raise Exception(f"Collection failed: {collection_result}")

        # Step 2: Validate data
        validation_result = validate_collected_data_task.apply(
            args=[collection_result["output_path"]]
        ).get()

        if not validation_result.get("validation", {}).get("passed", False):
            logger.warning("data_validation_failed_proceeding", action="proceed_with_caution")

        # Step 3: Optionally trigger training
        training_task_id = None
        if train_after_collection and validation_result.get("validation", {}).get("passed", False):
            from .ml_tasks import train_model_task

            train_result = train_model_task.apply_async()
            training_task_id = train_result.id
            logger.info("training_triggered", task_id=training_task_id)

        return {
            "task_id": self.request.id,
            "status": "success",
            "collection": collection_result,
            "validation": validation_result,
            "training_task_id": training_task_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("full_pipeline_failed", error=str(e))
        raise
