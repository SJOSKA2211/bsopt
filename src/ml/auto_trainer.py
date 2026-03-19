"""
ML Auto-Trainer for EquaFlow

Automatically triggers training when:
1. ML containers are healthy
2. DB is accessible with sufficient data
3. Training schedule triggers (daily or on-demand)

Coordinates with:
- Ray cluster for distributed training
- MLflow for experiment tracking
- Model registry for promotion
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class TrainingTrigger(Enum):
    """Training trigger types."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_DRIFT = "data_drift"
    MODEL_DEGRADATION = "model_degradation"
    NEW_DATA = "new_data"


class TrainingStatus(Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for training job."""

    model_name: str
    experiment_name: str
    symbols: list[str] = field(default_factory=list)
    market_type: str = "frontier"
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    trigger: TrainingTrigger = TrainingTrigger.SCHEDULED
    tags: dict[str, str] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of a training job."""

    status: TrainingStatus
    run_id: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    model_path: Optional[str] = None


class HealthChecker:
    """
    Checks health of dependencies:
    - Database (PostgreSQL)
    - Ray cluster
    - MLflow server
    - Redis cache
    """

    def __init__(self):
        self._db_healthy = False
        self._ray_healthy = False
        self._mlflow_healthy = False
        self._redis_healthy = False

    async def check_database(self) -> bool:
        """Check if database is accessible."""
        try:
            from src.database import db_manager

            db_manager.initialize()
            async with db_manager.session() as session:
                result = await session.execute("SELECT 1")
                self._db_healthy = result.scalar() == 1
                return self._db_healthy
        except Exception as e:
            logger.warning("database_health_check_failed", error=str(e))
            self._db_healthy = False
            return False

    async def check_ray(self) -> bool:
        """Check if Ray cluster is accessible."""
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            cluster_resources = ray.cluster_resources()
            available_cpu = cluster_resources.get("CPU", 0)
            available_gpu = cluster_resources.get("GPU", 0)

            self._ray_healthy = available_cpu > 0
            logger.info("ray_health_check", cpu=available_cpu, gpu=available_gpu)
            return self._ray_healthy
        except Exception as e:
            logger.warning("ray_health_check_failed", error=str(e))
            self._ray_healthy = False
            return False

    async def check_mlflow(self) -> bool:
        """Check if MLflow server is accessible."""
        try:
            import mlflow
            from src.shared.config import settings

            mlflow.set_tracking_uri(settings.tracking_uri)
            experiments = mlflow.list_experiments()
            self._mlflow_healthy = True
            logger.info("mlflow_health_check", experiments=len(experiments))
            return True
        except Exception as e:
            logger.warning("mlflow_health_check_failed", error=str(e))
            self._mlflow_healthy = False
            return False

    async def check_redis(self) -> bool:
        """Check if Redis is accessible."""
        try:
            from src.shared.utils.cache import get_redis_client

            redis = await get_redis_client()
            await redis.ping()
            self._redis_healthy = True
            return True
        except Exception as e:
            logger.warning("redis_health_check_failed", error=str(e))
            self._redis_healthy = False
            return False

    async def check_all(self) -> dict[str, bool]:
        """Check all dependencies."""
        results = {
            "database": await self.check_database(),
            "ray": await self.check_ray(),
            "mlflow": await self.check_mlflow(),
            "redis": await self.check_redis(),
        }
        return results

    @property
    def all_healthy(self) -> bool:
        """Check if all dependencies are healthy."""
        return all([self._db_healthy, self._ray_healthy, self._mlflow_healthy])


class AutoTrainer:
    """
    Orchestrates automated ML training pipeline.

    Responsibilities:
    - Health checking before training
    - Data ingestion from database
    - Ray-based distributed training
    - MLflow experiment tracking
    - Model registration and promotion
    """

    def __init__(
        self,
        check_interval: int = 60,
        training_interval: int = 86400,
    ):
        self.health_checker = HealthChecker()
        self.check_interval = check_interval
        self.training_interval = training_interval
        self._running = False
        self._last_training_time: Optional[float] = None
        self._current_result: Optional[TrainingResult] = None

    async def _get_available_symbols(self) -> list[str]:
        """Get list of available symbols from database."""
        try:
            from sqlalchemy import select, func
            from src.database.models import Symbol

            async with self.db_manager.session() as session:
                result = await session.execute(
                    select(Symbol.symbol, func.count(MarketTick.id))
                    .join(MarketTick, Symbol.symbol == MarketTick.symbol)
                    .group_by(Symbol.symbol)
                    .having(func.count(MarketTick.id) > 1000)
                )
                symbols = [row[0] for row in result.fetchall()]
                logger.info("symbols_loaded", count=len(symbols))
                return symbols
        except Exception as e:
            logger.warning("symbol_loading_failed", error=str(e))
            return []

    async def _prepare_data(
        self,
        symbols: list[str],
        market_type: str = "frontier",
    ) -> dict[str, Any]:
        """
        Prepare training data from database.

        Args:
            symbols: List of symbols to train on
            market_type: 'frontier' or 'emerging'

        Returns:
            Dictionary with training/validation/test splits
        """
        from sqlalchemy import select, text
        from src.database.models import MarketTick

        try:
            async with self.db_manager.session() as session:
                query = (
                    select(
                        MarketTick.symbol,
                        MarketTick.time,
                        MarketTick.price,
                        MarketTick.volume,
                    )
                    .where(MarketTick.symbol.in_(symbols))
                    .order_by(MarketTick.time)
                    .limit(1_000_000)
                )

                result = await session.execute(query)
                rows = result.fetchall()

                import pandas as pd

                df = pd.DataFrame(rows, columns=["symbol", "time", "price", "volume"])

                split_idx = int(len(df) * 0.8)
                train_idx = int(split_idx * 0.8)

                return {
                    "train": df.iloc[:train_idx].to_dict("records"),
                    "validation": df.iloc[train_idx:split_idx].to_dict("records"),
                    "test": df.iloc[split_idx:].to_dict("records"),
                    "metadata": {
                        "total_samples": len(df),
                        "symbols": symbols,
                        "market_type": market_type,
                    },
                }

        except Exception as e:
            logger.error("data_preparation_failed", error=str(e))
            return {"train": [], "validation": [], "test": [], "metadata": {}}

    async def _run_training(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute training job using Ray.

        Args:
            config: Training configuration

        Returns:
            Training result with metrics
        """
        result = TrainingResult(
            status=TrainingStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
        )

        try:
            import mlflow
            import ray
            from src.shared.config import settings

            mlflow.set_tracking_uri(settings.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

            with mlflow.start_run(tags=config.tags) as run:
                result.run_id = run.info.run_id

                mlflow.log_params(
                    {
                        "model_name": config.model_name,
                        "symbols": ",".join(config.symbols),
                        "market_type": config.market_type,
                        "epochs": config.epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "trigger": config.trigger.value,
                        **{k: str(v) for k, v in config.hyperparameters.items()},
                    }
                )

                from src.ml.distributed_training import BSOptDistributedTrainer

                trainer = BSOptDistributedTrainer()

                metrics = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: trainer.run(
                        {
                            "symbols": config.symbols,
                            "epochs": config.epochs,
                            "batch_size": config.batch_size,
                            "learning_rate": config.learning_rate,
                        }
                    ),
                )

                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                result.metrics = metrics

            result.status = TrainingStatus.COMPLETED
            result.end_time = datetime.now(timezone.utc)
            result.model_path = f"runs:/{result.run_id}/model"

            logger.info(
                "training_completed",
                run_id=result.run_id,
                metrics=metrics,
            )

        except Exception as e:
            result.status = TrainingStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now(timezone.utc)
            logger.error("training_failed", error=str(e))

        return result

    async def trigger_training(
        self,
        config: TrainingConfig,
    ) -> TrainingResult:
        """
        Trigger a new training job.

        Args:
            config: Training configuration

        Returns:
            Training result
        """
        logger.info(
            "training_triggered",
            model=config.model_name,
            trigger=config.trigger.value,
            symbols=config.symbols[:5] if config.symbols else [],
        )

        self._current_result = await self._run_training(config)

        if self._current_result.status == TrainingStatus.COMPLETED:
            from src.ml.evaluation.promote_and_rollback import ModelPromoter

            promoter = ModelPromoter(config.model_name)
            try:
                await promoter.promote_candidate(
                    self._current_result.run_id,
                )
            except Exception as e:
                logger.error("model_promotion_failed", error=str(e))

        return self._current_result

    async def _should_trigger_training(self) -> bool:
        """
        Determine if training should be triggered.

        Checks:
        - Time since last training
        - Available data
        - Training not already running
        """
        if self._current_result and self._current_result.status == TrainingStatus.RUNNING:
            return False

        if self._last_training_time:
            elapsed = time.time() - self._last_training_time
            if elapsed < self.training_interval:
                logger.debug(
                    "training_cooldown_active",
                    remaining=self.training_interval - elapsed,
                )
                return False

        return True

    async def run(self) -> None:
        """
        Main training loop.

        Continuously monitors health and triggers training when appropriate.
        """
        self._running = True
        logger.info("auto_trainer_started", check_interval=self.check_interval)

        while self._running:
            try:
                health = await self.health_checker.check_all()

                if not health["database"]:
                    logger.warning("database_unhealthy_skipping_training_cycle")
                    await asyncio.sleep(self.check_interval)
                    continue

                if not health["ray"]:
                    logger.warning("ray_unhealthy_skipping_training_cycle")
                    await asyncio.sleep(self.check_interval)
                    continue

                if not self.health_checker.all_healthy:
                    logger.warning(
                        "dependencies_unhealthy",
                        health=health,
                    )
                    await asyncio.sleep(self.check_interval)
                    continue

                if await self._should_trigger_training():
                    symbols = await self._get_available_symbols()

                    config = TrainingConfig(
                        model_name="equity_pricing_v1",
                        experiment_name="equity_pricing",
                        symbols=symbols[:50],
                        market_type="frontier",
                        epochs=100,
                        batch_size=256,
                        trigger=TrainingTrigger.SCHEDULED,
                    )

                    result = await self.trigger_training(config)
                    self._last_training_time = time.time()

                    if result.status == TrainingStatus.COMPLETED:
                        logger.info(
                            "training_cycle_complete",
                            duration=(result.end_time - result.start_time).total_seconds(),
                        )
                    else:
                        logger.error(
                            "training_cycle_failed",
                            error=result.error,
                        )

            except Exception as e:
                logger.error("training_loop_error", error=str(e))

            await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop the training loop."""
        self._running = False
        logger.info("auto_trainer_stopped")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="EquaFlow Auto Trainer")
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Health check interval in seconds",
    )
    parser.add_argument(
        "--training-interval",
        type=int,
        default=86400,
        help="Minimum interval between training runs in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    args = parser.parse_args()

    trainer = AutoTrainer(
        check_interval=args.check_interval,
        training_interval=args.training_interval,
    )

    if args.once:
        health = await trainer.health_checker.check_all()
        print(f"Health Check Results: {health}")

        if all(health.values()):
            symbols = await trainer._get_available_symbols()
            config = TrainingConfig(
                model_name="equity_pricing_v1",
                experiment_name="equity_pricing",
                symbols=symbols[:10],
                trigger=TrainingTrigger.MANUAL,
            )
            result = await trainer.trigger_training(config)
            print(f"Training Result: {result}")
        else:
            print("Health checks failed. Cannot start training.")
    else:
        await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
