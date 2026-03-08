"""
Unified Advanced ML Pipeline for BS-OPT

Consolidated pipeline for data ingestion, feature engineering, distributed HPO (Ray),
temporal validation, and automated model promotion.
"""

import asyncio
import os
import re
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import ray
import structlog
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sqlalchemy import func, select

from src.config import get_settings
from src.database import dispose_engine, get_async_db_context, get_engine
from src.database.models import ModelPrediction
from src.ml.drift import DriftTrigger, PerformanceDriftMonitor
from src.ml.indicators import (
    get_adx,
    get_atr,
    get_bbands,
    get_macd,
    get_rsi,
)
from src.ml.scraper import MarketDataScraper
from src.ml.trainer import ModelTrainer
from src.shared.observability import push_metrics, setup_logging

logger = structlog.get_logger(__name__)


class MLPipeline:
    """
    The Single Source of Truth for ML at BS-OPT.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        setup_logging()
        self.settings = get_settings()
        self.config = config or {}

        # 1. Validate Inputs (Security & QA)
        self.ticker = str(self.config.get("ticker", "AAPL")).upper()
        if not re.match(r"^[A-Z0-9.-]{1,20}$", self.ticker):
            raise ValueError(f"Invalid ticker: {self.ticker}")

        self.study_name = str(self.config.get("study_name", f"god_mode_{self.ticker.lower()}"))
        if not re.match(r"^[a-zA-Z0-9_\-]{1,100}$", self.study_name):
            raise ValueError(f"Invalid study_name: {self.study_name}")

        # MLflow Config
        self.tracking_uri = self.config.get("tracking_uri", self.settings.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

        # Components
        self.scraper = MarketDataScraper(
            api_key=self.config.get("api_key", os.getenv("ALPHA_VANTAGE_API_KEY", "DEMO_KEY")),
            provider=self.config.get("provider", "auto"),
        )
        self.framework = self.config.get("framework", "xgboost")

        # Database
        self.engine = get_engine()

        self.drift_trigger = DriftTrigger(self.config)
        self.performance_monitor = PerformanceDriftMonitor()

    async def shutdown(self):
        """Gracefully shutdown the pipeline and dispose of engines."""
        await dispose_engine()
        logger.info("ml_pipeline_shutdown_complete")

    async def get_current_model_performance(self, session) -> float | None:
        """Fetches the average accuracy of the current model from recent predictions."""
        try:
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
        """Vectorized feature engineering using pricing kernels."""
        df = df.sort_values("timestamp").reset_index(drop=True)
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)

        # Returns
        df["log_return"] = np.concatenate([[0], np.log(closes[1:] / closes[:-1])])

        # Indicators
        df["RSI_14"] = get_rsi(closes, length=14)
        macd, signal, _ = get_macd(closes)
        df["MACD"] = macd
        df["MACD_Signal"] = signal

        lower, mid, upper = get_bbands(closes)
        df["BBL"] = lower
        df["BBM"] = mid
        df["BBU"] = upper

        df["ATR_14"] = get_atr(highs, lows, closes)
        df["ADX_14"] = get_adx(highs, lows, closes)

        return df.dropna().copy()

    async def run(self, force: bool = False):
        """Execute the full ML pipeline loop."""
        logger.info("pipeline_initiated", ticker=self.ticker)

        # 1. Fetch & Persist
        try:
            df = await self._fetch_data()
            await self._persist_data(df)
        except Exception as e:
            if self.config.get("use_synthetic_fallback", True):
                logger.warning("data_fetch_failed_using_synthetic_fallback", error=str(e))
                df = await self._get_synthetic_data()
            else:
                raise e

        df_featured = self.generate_features(df)

        # 2. Advanced Drift Check
        if not force:
            async with get_async_db_context() as session:
                current_perf = await self.get_current_model_performance(session)

            mid = len(df_featured) // 2
            reference_df = df_featured.iloc[:mid]
            current_df = df_featured.iloc[mid:]

            # A. Univariate Drift (Target Variable)
            should_retrain_uni, reason_uni = self.drift_trigger.should_retrain(
                reference_df["log_return"].values, 
                current_df["log_return"].values, 
                current_perf
            )

            # B. Multivariate Drift (Feature Set)
            from src.ml.monitoring.mmd import MultivariateDriftDetector
            
            features_to_monitor = [
                col for col in df_featured.columns 
                if col not in ["timestamp", "ticker", "log_return"]
            ]
            
            mmd_detector = MultivariateDriftDetector(
                threshold=self.config.get("mmd_threshold", 0.08)
            )
            
            is_drifted_multi, mmd_score = mmd_detector.detect_drift(
                reference_df[features_to_monitor].values,
                current_df[features_to_monitor].values
            )

            # C. Aggregate Decision
            if not(should_retrain_uni or is_drifted_multi):
                logger.info("pipeline_skipped_no_drift", mmd_score=mmd_score)
                return None
            
            reason = reason_uni if should_retrain_uni else "multivariate_feature_drift"
            logger.warning("drift_detected_triggering_retrain", reason=reason, mmd=mmd_score)

        # 3. Distributed HPO via Ray
        best_config = self._run_distributed_hpo(df_featured)

        # 4. Final Train & Promote
        trainer = ModelTrainer(self.study_name, tracking_uri=self.tracking_uri)
        x_vals, y_vals, features, meta = self._prepare_data(df_featured)

        avg_score = trainer.train_and_evaluate(
            x_vals, y_vals, best_config, feature_names=features, dataset_metadata=meta
        )

        promote_threshold = self.config.get(
            "promote_threshold", self.settings.ML_TRAINING_PROMOTE_THRESHOLD_R2
        )
        if avg_score > promote_threshold:
            self._promote(trainer.model, avg_score)

        push_metrics(job_name="unified_ml_pipeline")
        return trainer.model

    def _run_distributed_hpo(self, df: pd.DataFrame) -> dict[str, Any]:
        """HPO powered by Ray Tune and Optuna."""
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "DATABASE_URL": self.settings.DATABASE_URL,
                        "REDIS_URL": self.settings.REDIS_URL,
                        "JWT_SECRET": self.settings.JWT_SECRET,
                        "INSIDE_DOCKER": "1",
                    }
                },
            )

        framework = self.config.get("framework", "xgboost")

        if framework == "xgboost":
            search_space = {
                "n_estimators": tune.randint(50, 500),
                "max_depth": tune.randint(3, 12),
                "learning_rate": tune.loguniform(1e-3, 0.3),
                "framework": "xgboost",
            }
        elif framework == "nn":
            search_space = {
                "lr": tune.loguniform(1e-4, 1e-2),
                "epochs": tune.choice([50, 100, 200]),
                "batch_size": tune.choice([16, 32, 64]),
                "framework": "nn",
            }
        else:
            search_space = {"framework": framework}

        # Prepare data once to avoid repeated overhead and capture
        x, y, features, meta = self._prepare_data(df)
        study_name = self.study_name

        def train_func(config):
            # Static instantiation inside the worker
            from src.ml.trainer import ModelTrainer

            trainer = ModelTrainer(study_name)
            score = trainer.train_and_evaluate(x, y, config)
            tune.report(mean_score=score)

        algo = OptunaSearch()
        tuner = tune.Tuner(
            train_func,
            tune_config=tune.TuneConfig(
                metric="mean_score",
                mode="max",
                search_alg=algo,
                num_samples=self.config.get("n_trials", 10),
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        return results.get_best_result().config

    def _prepare_data(self, df: pd.DataFrame):
        """Prepare tensors for training."""
        df["target"] = df["log_return"].shift(-1)
        df = df.iloc[:-1]
        features = [
            str(col) for col in df.columns if str(col) not in ["timestamp", "target", "ticker"]
        ]

        # Scaling for NN
        if self.config.get("framework") == "nn":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X = scaler.fit_transform(df[features].values)
        else:
            X = df[features].values

        return (
            X,
            df["target"].values,
            features,
            {"ticker": self.ticker, "framework": self.framework},
        )

    async def _fetch_data(self) -> pd.DataFrame:
        """Ingest historical data."""
        from datetime import date, timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=self.config.get("trailing_days", 365))
        return await self.scraper.fetch_historical_data(
            self.ticker, start_date.isoformat(), end_date.isoformat()
        )

    async def _get_synthetic_data(self) -> pd.DataFrame:
        """Generates synthetic fallback data."""
        from src.ml.training.data_gen import generate_synthetic_data_numba

        n_samples = self.config.get("synthetic_samples", 1000)
        x, y, cols = generate_synthetic_data_numba(n_samples=n_samples)

        # Map to DataFrame structure expected by generate_features
        df = pd.DataFrame(x, columns=cols)
        df["close"] = y  # In synthetic, y is typically the price or target
        df["timestamp"] = np.arange(len(df))
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        return df

    async def _persist_data(self, df: pd.DataFrame):
        """Internal helper to persist data."""
        from src.database.crud import bulk_insert_market_ticks

        market_data_records = [
            {
                "time": pd.to_datetime(ts, unit="s", utc=True)
                if isinstance(ts, int | float)
                else pd.to_datetime(ts),
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

    def _promote(self, model, score):
        """Register model in MLflow."""
        framework_tag = self.config.get("framework", self.framework)
        name = f"PRICER_{self.ticker}_{framework_tag}".upper()
        run = mlflow.active_run()
        if run:
            try:
                result = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=name, version=result.version, stage="Production"
                )
                logger.info("model_promoted", name=name, score=score, version=result.version)
            except Exception as e:
                logger.error("mlflow_promotion_failed", error=str(e))

            # Async ONNX Export
            model_path = f"models/{self.study_name}_latest.onnx"
            quantized_path = f"models/{self.study_name}_latest.int8.onnx"
            try:
                from src.tasks.ml_tasks import optimize_model_task

                optimize_model_task.delay(model_path, quantized_path)
            except Exception as e:
                logger.error("onnx_export_failed", error=str(e))


if __name__ == "__main__":
    p = MLPipeline({"ticker": "TSLA", "n_trials": 2})
    asyncio.run(p.run(force=True))
