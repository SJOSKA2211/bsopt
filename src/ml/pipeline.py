"""
Unified Advanced ML Pipeline for BS-OPT
=======================================

Consolidated pipeline for data ingestion, feature engineering, distributed HPO (Ray),
temporal validation, and automated model promotion.
"""

import asyncio
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import ray
import structlog
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sqlalchemy import create_engine

from src.config import get_settings
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

        # MLflow Config
        self.tracking_uri = self.config.get("tracking_uri", self.settings.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

        # Components
        self.scraper = MarketDataScraper(
            api_key=self.config.get("api_key", os.getenv("ALPHA_VANTAGE_API_KEY", "DEMO_KEY")),
            provider=self.config.get("provider", "auto"),
        )
        self.ticker = self.config.get("ticker", "AAPL")
        self.study_name = self.config.get("study_name", f"god_mode_{self.ticker.lower()}")
        self.framework = self.config.get("framework", "xgboost")

        # Database
        self.engine = create_engine(self.settings.DATABASE_URL)

        self.drift_trigger = DriftTrigger(self.config)
        self.performance_monitor = PerformanceDriftMonitor()

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
        """Execute the god-mode loop."""
        logger.info("pipeline_initiated", ticker=self.ticker)

        # 1. Fetch & Persist
        df = await self._fetch_data()
        df_featured = self.generate_features(df)

        # 2. Drift Check
        if not force:
            should_retrain, _ = self.drift_trigger.should_retrain(df_featured["close"].values)
            if not should_retrain:
                logger.info("pipeline_skipped_no_drift")
                return None

        # 3. Distributed HPO via Ray
        best_config = self._run_distributed_hpo(df_featured)

        # 4. Final Train & Promote
        trainer = ModelTrainer(self.study_name, tracking_uri=self.tracking_uri)
        x_vals, y_vals, features, meta = self._prepare_data(df_featured)

        avg_r2 = trainer.train_and_evaluate(
            x_vals, y_vals, best_config, feature_names=features, dataset_metadata=meta
        )

        if avg_r2 > self.settings.ML_TRAINING_PROMOTE_THRESHOLD_R2:
            self._promote(trainer.model, avg_r2)

        push_metrics(job_name="unified_ml_pipeline")
        return trainer.model

    def _run_distributed_hpo(self, df: pd.DataFrame) -> dict[str, Any]:
        """HPO powered by Ray Tune and Optuna."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        search_space = {
            "n_estimators": tune.randint(50, 500),
            "max_depth": tune.randint(3, 12),
            "learning_rate": tune.loguniform(1e-3, 0.3),
            "framework": tune.choice(["xgboost"]),
        }

        def train_func(config):
            trainer = ModelTrainer(self.study_name)
            x, y, _, _ = self._prepare_data(df)
            score = trainer.train_and_evaluate(x, y, config)
            tune.report(mean_r2=score)

        algo = OptunaSearch()
        tuner = tune.Tuner(
            train_func,
            tune_config=tune.TuneConfig(
                metric="mean_r2",
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
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.iloc[:-1]
        features = [
            c for col in df.columns if (c := str(col)) not in ["timestamp", "target", "ticker"]
        ]
        return (
            df[features].values,
            df["target"].values,
            features,
            {"ticker": self.ticker},
        )

    async def _fetch_data(self) -> pd.DataFrame:
        """Ingest historical data."""
        return await self.scraper.fetch_historical_data(self.ticker, "2025-01-01", "2026-02-01")

    def _promote(self, model, score):
        """Register model in MLflow."""
        name = f"PRICER_{self.ticker}_{self.framework}".upper()
        run = mlflow.active_run()
        if run:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
            logger.info("model_promoted", name=name, score=score)


if __name__ == "__main__":
    p = MLPipeline({"ticker": "TSLA", "n_trials": 2})
    asyncio.run(p.run(force=True))
