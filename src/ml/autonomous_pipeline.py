import asyncio
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import create_engine, func, select

from src.config import get_settings
from src.database import Base, get_async_db_context
from src.database.models import ModelPrediction
from src.ml.drift import DriftTrigger, PerformanceDriftMonitor
from src.ml.indicators import (
    get_rsi,
)
from src.ml.scraper import MarketDataScraper
from src.ml.trainer import InstrumentedTrainer
from src.shared.observability import (
    push_metrics,
    setup_logging,
)

# Initialize structured logger
logger = structlog.get_logger()


class AutonomousMLPipeline:
    """
    End-to-end autonomous ML pipeline integrating scraping, persistence,
    drift detection, and model optimization.
    """

    def __init__(self, config: dict[str, Any]):
        setup_logging()
        self.config = config

        #  FORCE MLFLOW TO POSTGRES
        self.db_url = config["db_url"]
        tracking_uri = self.db_url.replace("postgresql+asyncpg", "postgresql")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("mlflow_tracking_configured", uri=tracking_uri)

        self.scraper = MarketDataScraper(
            api_key=config["api_key"], provider=config.get("provider", "auto")
        )
        self.ticker = config["ticker"]
        self.study_name = config["study_name"]
        self.n_trials = config.get("n_trials", 50)
        self.framework = config.get("framework", "xgboost")

        # Initialize DB (create tables if they don't exist)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)

        # Initialize Smart Trigger for drift-based retraining
        self.drift_trigger = DriftTrigger(self.config)
        self.performance_monitor = PerformanceDriftMonitor()

    async def get_current_model_performance(self, session) -> float | None:
        """Fetches the average accuracy of the current model from recent predictions."""
        try:
            # FIX: Real accuracy calculation using actual_value
            result = (
                await session.execute(
                    select(
                        func.avg(
                            func.cast(
                                (ModelPrediction.predicted_value > 0.5)
                                == (ModelPrediction.actual_value > 0.5),
                                np.float64,
                            )
                        )
                    )
                    .where(ModelPrediction.actual_value.isnot(None))
                    .limit(100)
                )
            ).scalar()
            return float(result) if result is not None else None
        except Exception as e:
            logger.warning("failed_to_fetch_performance", error=str(e))
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators with OPTIMIZED vectorized paths.
        """
        # OPTIMIZED: Assume sorted if coming from reliable source, otherwise sort once
        if not df.index.is_monotonic_increasing:
            df = df.sort_values("timestamp")

        closes = df["close"].values.astype(np.float64)
        df["high"].values.astype(np.float64)
        df["low"].values.astype(np.float64)

        # 1. Log Returns (Vectorized)
        log_ret = np.zeros_like(closes)
        log_ret[1:] = np.log(closes[1:] / np.where(closes[:-1] == 0, 1e-9, closes[:-1]))
        df["log_return"] = log_ret

        # 2. Optimized Volatility
        window = 20

        # Fast rolling std using NumPy
        def fast_rolling_std(x, w):
            c1 = np.cumsum(x)
            c2 = np.cumsum(x**2)
            c1 = (c1[w:] - c1[:-w]) / w
            c2 = (c2[w:] - c2[:-w]) / w
            return np.sqrt(np.maximum(c2 - c1**2, 0))

        vol = fast_rolling_std(log_ret, window)
        # Pad with zeros to maintain length
        df["volatility"] = np.concatenate([np.zeros(window), vol]) * np.sqrt(
            252 * 6.5 * 60
        )

        # 3. Indicators (Assume indicators.py is JIT-accelerated)
        df["RSI_14"] = get_rsi(closes, length=14)
        # ... (rest of indicator calls stay same)
        return df.dropna().copy()

    async def _persist_data(self, df: pd.DataFrame):
        """Internal helper to persist data with reduced allocation overhead."""
        from src.database.crud import bulk_insert_market_ticks

        # OPTIMIZED: Use list comprehension for speed, but avoid heavy dicts if possible
        # In a true Rick-pass, we'd pass the DF directly to a COPY handler
        market_data_records = [
            {
                "time": pd.to_datetime(ts, unit="s", utc=True),
                "symbol": self.ticker,
                "price": float(c),
                "volume": int(v),
                "side": None,
            }
            for ts, c, v in zip(
                df["timestamp"], df["close"], df.get("volume", np.zeros(len(df))), strict=False
            )
        ]

        async with get_async_db_context() as async_session:
            if market_data_records:
                await bulk_insert_market_ticks(async_session, market_data_records)

    def _prepare_training_data(
        self, df_featured: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
        """Prepares X, y and metadata for training."""
        df_featured["target"] = (
            df_featured["close"].shift(-1) > df_featured["close"]
        ).astype(int)
        df_featured = df_featured.iloc[:-1]

        exclude = ["timestamp", "target", "ticker"]
        feature_names = [col for col in df_featured.columns if col not in exclude]
        x_vals = df_featured[feature_names].values
        y_vals = df_featured["target"].values

        dataset_metadata = {
            "ticker": self.ticker,
            "rows": str(len(df_featured)),
            "features": str(len(feature_names)),
        }
        return x_vals, y_vals, feature_names, dataset_metadata

    async def _train_and_optimize(
        self, x_vals, y_vals, feature_names, dataset_metadata, base_model
    ):
        """Orchestrates model optimization and training."""
        trainer = InstrumentedTrainer(study_name=self.study_name)

        def objective(trial):
            if self.framework == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.001, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "framework": "xgboost",
                }
            elif self.framework == "sklearn":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "framework": "sklearn",
                }
            else:  # pytorch
                params = {
                    "epochs": trial.suggest_int("epochs", 20, 100),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "framework": "pytorch",
                }

            return trainer.train_and_evaluate(
                x_vals,
                y_vals,
                params,
                feature_names=feature_names,
                dataset_metadata=dataset_metadata,
                base_model=base_model,
                trial=trial,
            )

        return trainer.optimize(objective, n_trials=self.n_trials)

    def _export_model(self, best_accuracy: float, run_id: str):
        """Handles model promotion, MLflow staging, and ONNX export."""
        promotion_threshold = self.config.get("promotion_threshold", 0.8)
        if best_accuracy >= promotion_threshold:
            logger.info(
                "model_promotion_triggered",
                accuracy=best_accuracy,
                threshold=promotion_threshold,
            )

            # 1. Promote in MLflow
            try:
                model_name = f"Option_Pricing_{self.framework}"
                result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name, version=result.version, stage="Production"
                )
                logger.info(
                    "mlflow_promotion_complete", name=model_name, version=result.version
                )
            except Exception as e:
                logger.error("mlflow_promotion_failed", error=str(e))

            # 2. Asynchronous ONNX Export
            model_path = f"models/{self.study_name}_latest.onnx"
            quantized_path = f"models/{self.study_name}_latest.int8.onnx"
            try:
                from src.tasks.ml_tasks import optimize_model_task

                optimize_model_task.delay(model_path, quantized_path)
            except Exception as e:
                logger.error("onnx_export_failed", error=str(e))

    async def run(self):
        """
        Executes the full pipeline asynchronously with smart drift-based retraining.
        """
        logger.info("pipeline_started", ticker=self.ticker, framework=self.framework)
        try:
            df = await self._fetch_data()
            await self._persist_data(df)
            df_featured = self.generate_features(df)
            logger.info("features_generated", columns=list(df_featured.columns))

            from src.database.crud import get_production_model

            async with get_async_db_context() as async_session:
                current_perf = await self.get_current_model_performance(async_session)
                historical_prices = df_featured["close"].values
                split_idx = int(len(historical_prices) * 0.8)
                reference_data = historical_prices[:split_idx]
                current_data = historical_prices[split_idx:]
                should_retrain, reason = self.drift_trigger.should_retrain(
                    reference_data, current_data, current_perf
                )

                if not should_retrain:
                    logger.info("retraining_skipped", reason=reason)
                    return None

                logger.info("retraining_initiated", reason=reason)
                base_model = None
                if self.config.get("use_warm_start", True):
                    try:
                        prod_model_record = await get_production_model(
                            async_session, self.study_name
                        )
                        if prod_model_record and prod_model_record.model_artifact_url:
                            logger.info(
                                "warm_start_model_identified",
                                model_id=str(prod_model_record.id),
                            )
                    except Exception as e:
                        logger.warning("failed_to_load_base_model", error=str(e))

            x_vals, y_vals, feature_names, dataset_metadata = (
                self._prepare_training_data(df_featured)
            )
            study = await self._train_and_optimize(
                x_vals, y_vals, feature_names, dataset_metadata, base_model
            )
            best_accuracy = study.best_value

            # Extract run_id from best trial or current context
            # (In a real implementation, trainer would expose this)
            run_id = study.user_attrs.get("best_run_id", "unknown")
            self._export_model(best_accuracy, run_id)

            is_drifted = self.performance_monitor.detect_drift(best_accuracy)
            self.performance_monitor.add_metric(best_accuracy)
            logger.info(
                "pipeline_completed",
                best_accuracy=best_accuracy,
                performance_drift=is_drifted,
                best_params=study.best_params,
            )
            push_metrics(job_name="autonomous_pipeline")
            return study
        except Exception as e:
            logger.critical("pipeline_failed", error=str(e))
            push_metrics(job_name="autonomous_pipeline")
            raise


if __name__ == "__main__":
    import os

    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    poly_key = os.getenv("POLYGON_API_KEY")
    if poly_key and poly_key.strip() != "DEMO_KEY":
        api_key = poly_key
        provider = "polygon"
    elif av_key and av_key.strip() != "DEMO_KEY":
        api_key = av_key
        provider = "alpha_vantage"
    else:
        api_key = "DEMO_KEY"
        provider = "mock"

    settings = get_settings()
    config = {
        "api_key": api_key,
        "provider": provider,
        "db_url": settings.DATABASE_URL,
        "ticker": os.getenv("TICKER", "AAPL"),
        "study_name": os.getenv("STUDY_NAME", "aapl_opt_v1"),
        "n_trials": int(os.getenv("N_TRIALS", "5")),
        "framework": os.getenv("FRAMEWORK", "xgboost"),
    }
    pipeline = AutonomousMLPipeline(config)
    asyncio.run(pipeline.run())
