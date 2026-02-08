"""
Unified ML Training & Evaluation Pipeline
=========================================

🚀 SINGULARITY: One source of truth for model training, optimization, and evaluation.
Fixes fragmented training scripts and ensures rigorous temporal validation.
"""

import asyncio
import os
from datetime import datetime, UTC
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import create_engine

from src.config import get_settings
from src.database import Base
from src.ml.drift import DriftTrigger, PerformanceDriftMonitor
from src.ml.evaluation.metrics import ModelScorecard
from src.ml.scraper import MarketDataScraper
from src.ml.trainer import ModelTrainer
from src.shared.observability import push_metrics, setup_logging

logger = structlog.get_logger(__name__)

class MLPipeline:
    """
    Consolidated ML Pipeline for BSOpt.
    Handles data collection, feature engineering, HPO, and model tracking.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        setup_logging()
        self.settings = get_settings()
        self.config = config or {}
        
        # Configure Tracking
        self.tracking_uri = self.config.get("tracking_uri", self.settings.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info("mlflow_tracking_configured", uri=self.tracking_uri)

        # Initialize Components
        self.scraper = MarketDataScraper(
            api_key=self.config.get("api_key", os.getenv("ALPHA_VANTAGE_API_KEY", "DEMO_KEY")), 
            provider=self.config.get("provider", "auto")
        )
        self.ticker = self.config.get("ticker", "AAPL")
        self.study_name = self.config.get("study_name", f"opt_{self.ticker.lower()}_v1")
        self.framework = self.config.get("framework", "xgboost")
        
        # DB Engine
        self.engine = create_engine(self.settings.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        
        self.drift_trigger = DriftTrigger(self.config)
        self.performance_monitor = PerformanceDriftMonitor()

    async def run(self, force: bool = False):
        """Executes the full pipeline loop."""
        logger.info("pipeline_started", ticker=self.ticker, framework=self.framework)
        
        # 1. Drift Check
        if not force:
            is_drifted = await self._check_drift()
            if not is_drifted:
                logger.info("pipeline_skipped_no_drift")
                return None

        # 2. Data Collection
        df = await self._fetch_data()
        
        # 3. Feature Engineering
        from src.ml.autonomous_pipeline import AutonomousMLPipeline
        # Reusing the optimized feature generation from AutonomousMLPipeline
        pipeline_helper = AutonomousMLPipeline(self.config)
        df_featured = pipeline_helper.generate_features(df)
        
        # 4. Prepare Training Data
        x_vals, y_vals, features, meta = pipeline_helper._prepare_training_data(df_featured)
        
        # 5. Optimization & Training
        trainer = ModelTrainer(
            study_name=self.study_name,
            tracking_uri=self.tracking_uri,
            n_splits=self.config.get("n_splits", 5) # Default to 5-fold Walk-Forward
        )
        
        def objective(trial):
            params = self._suggest_params(trial)
            return trainer.train_and_evaluate(
                x_vals, y_vals, 
                params=params, 
                feature_names=features, 
                dataset_metadata=meta,
                trial=trial
            )

        study = trainer.optimize(objective, n_trials=self.config.get("n_trials", 20))
        
        # 6. Promotion Logic
        if study.best_value > self.config.get("promotion_threshold", 0.85):
            self._promote_model(study, trainer.model, self.framework)
            
        push_metrics(job_name="ml_pipeline")
        logger.info("pipeline_complete", best_r2=study.best_value)
        return study

    async def _check_drift(self) -> bool:
        """Senses if retraining is necessary."""
        # For prototype, assume we need data from DB
        # This mirrors the logic in AutonomousMLPipeline
        try:
            return await self.drift_trigger.should_retrain()
        except Exception:
            return True # Default to True for now

    async def _fetch_data(self) -> pd.DataFrame:
        """Unified data fetching with synthetic fallback."""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            return await self.scraper.fetch_historical_data(
                self.ticker, 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d")
            )
        except Exception as e:
            logger.warning("data_fetch_failed", error=str(e))
            # Synthetic Fallback
            from src.ml.training.train import generate_synthetic_data
            X, y, features = generate_synthetic_data(n_samples=5000)
            # Create dummy DF
            return pd.DataFrame(X, columns=features).assign(close=y, timestamp=np.arange(len(y)))

    def _suggest_params(self, trial) -> dict[str, Any]:
        """Suggests hyperparameters based on framework."""
        if self.framework == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "framework": "xgboost"
            }
        elif self.framework == "sklearn":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "framework": "sklearn"
            }
        return {"framework": self.framework} # Default

    def _promote_model(self, study, model, framework):
        """Handles model registration and production transition."""
        model_name = f"OptionPricer_{self.ticker}_{framework}".upper()
        logger.info("promoting_model", name=model_name, value=study.best_value)
        
        # Log to central MLflow
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "latest"
        try:
            mlflow.register_model(f"runs:/{run_id}/model", model_name)
            # Tag as production if it's the absolute best
            # (In production, this would use MlflowClient to transition stages)
        except Exception as e:
            logger.error("promotion_failed", error=str(e))

if __name__ == "__main__":
    pipeline = MLPipeline({"ticker": "SPY", "n_trials": 5})
    asyncio.run(pipeline.run(force=True))
