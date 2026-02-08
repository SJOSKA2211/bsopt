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
    get_adx,
    get_atr,
    get_bbands,
    get_macd,
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
        
        # 🚀 FORCE MLFLOW TO POSTGRES
        self.db_url = config["db_url"]
        tracking_uri = self.db_url.replace("postgresql+asyncpg", "postgresql")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("mlflow_tracking_configured", uri=tracking_uri)

        self.scraper = MarketDataScraper(
            api_key=config["api_key"], 
            provider=config.get("provider", "auto")
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
            result = (await session.execute(
                select(func.avg(
                    func.cast(
                        (ModelPrediction.predicted_value > 0.5) == (ModelPrediction.actual_value > 0.5),
                        np.float64
                    )
                ))
                .where(ModelPrediction.actual_value.isnot(None))
                .limit(100)
            )).scalar()
            return float(result) if result is not None else None
        except Exception as e:
            logger.warning("failed_to_fetch_performance", error=str(e))
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators. Uses kernels from indicators.py.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        
        log_ret = np.log(closes[1:] / closes[:-1], where=(closes[:-1] != 0), out=np.zeros_like(closes[1:]))
        pct_ret = (closes[1:] / closes[:-1]) - 1
        
        df["log_return"] = np.concatenate([np.zeros(1), log_ret])
        df["pct_return"] = np.concatenate([np.zeros(1), pct_ret])
        
        window = 20
        returns = df["pct_return"].values
        def rolling_std(x, w):
            s1 = np.convolve(x, np.ones(w), 'valid')
            s2 = np.convolve(x**2, np.ones(w), 'valid')
            return np.sqrt((s2 - s1**2 / w) / (w - 1))
        
        vol = rolling_std(returns, window)
        df["volatility"] = np.concatenate([np.full(window - 1, np.nan), vol]) * np.sqrt(252 * 6.5 * 60)
        
        df["RSI_14"] = get_rsi(closes, length=14)
        macd, signal, hist = get_macd(closes, fast=12, slow=26, signal=9)
        df["MACD_12_26_9"] = macd
        df["MACDs_12_26_9"] = signal
        df["MACDh_12_26_9"] = hist
        lower, mid, upper = get_bbands(closes, length=20, num_std=2.0)
        df["BBL_20_2.0"] = lower
        df["BBM_20_2.0"] = mid
        df["BBU_20_2.0"] = upper
        df["ATR_14"] = get_atr(highs, lows, closes, length=14)
        df["ADX_14"] = get_adx(highs, lows, closes, length=14)
        
        return df.dropna().copy()

    async def _fetch_data(self) -> pd.DataFrame:
        """Internal helper to fetch market data."""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        str_start = start_date.strftime("%Y-%m-%d")
        str_end = end_date.strftime("%Y-%m-%d")
        
        try:
            df = await self.scraper.fetch_historical_data(self.ticker, str_start, str_end)
        except Exception as e:
            logger.warning("scrape_failed_retrying_mock", error=str(e))
            self.scraper = MarketDataScraper(api_key="DEMO_KEY", provider="mock")
            df = await self.scraper.fetch_historical_data(self.ticker, str_start, str_end)

        logger.info("data_scraped", rows=len(df))
        return df

    async def _persist_data(self, df: pd.DataFrame):
        """Internal helper to persist data to database."""
        from src.database.crud import bulk_insert_market_ticks
        market_data_records = [
            {
                "time": pd.to_datetime(ts, unit='s', utc=True),
                "symbol": self.ticker,
                "price": float(close),
                "volume": int(vol),
                "side": None
            }
            for ts, close, vol in zip(df["timestamp"], df["close"], df.get("volume", [0]*len(df)))
        ]
        
        async with get_async_db_context() as async_session:
            if market_data_records:
                await bulk_insert_market_ticks(async_session, market_data_records)

    def _prepare_training_data(self, df_featured: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
        """Prepares X, y and metadata for training."""
        df_featured["target"] = (df_featured["close"].shift(-1) > df_featured["close"]).astype(int)
        df_featured = df_featured.iloc[:-1]
        
        exclude = ["timestamp", "target", "ticker"]
        feature_names = [col for col in df_featured.columns if col not in exclude]
        x_vals = df_featured[feature_names].values
        y_vals = df_featured["target"].values
        
        dataset_metadata = {
            "ticker": self.ticker,
            "rows": str(len(df_featured)),
            "features": str(len(feature_names))
        }
        return x_vals, y_vals, feature_names, dataset_metadata

    async def _train_and_optimize(self, x_vals, y_vals, feature_names, dataset_metadata, base_model):
        """Orchestrates model optimization and training."""
        trainer = InstrumentedTrainer(study_name=self.study_name)
        
        def objective(trial):
            if self.framework == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "framework": "xgboost"
                }
            elif self.framework == "sklearn":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "framework": "sklearn"
                }
            else: # pytorch
                params = {
                    "epochs": trial.suggest_int("epochs", 20, 100),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "framework": "pytorch"
                }
            
            return trainer.train_and_evaluate(
                x_vals, y_vals, params, 
                feature_names=feature_names, 
                dataset_metadata=dataset_metadata,
                base_model=base_model,
                trial=trial
            )
        
        return trainer.optimize(objective, n_trials=self.n_trials)

    def _export_model(self, best_accuracy: float):
        """Handles model promotion and ONNX export."""
        if best_accuracy >= self.config.get("promotion_threshold", 0.8):
            logger.info("model_promotion_triggered", accuracy=best_accuracy)
            model_path = f"models/{self.study_name}_latest.onnx"
            quantized_path = f"models/{self.study_name}_latest.int8.onnx"
            try:
                logger.info("exporting_model_to_onnx", path=model_path)
                from src.tasks.ml_tasks import optimize_model_task
                optimize_model_task.delay(model_path, quantized_path)
            except Exception as e:
                logger.error("model_export_failed", error=str(e))

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
                should_retrain, reason = self.drift_trigger.should_retrain(reference_data, current_data, current_perf)

                if not should_retrain:
                    logger.info("retraining_skipped", reason=reason)
                    return None

                logger.info("retraining_initiated", reason=reason)
                base_model = None
                if self.config.get("use_warm_start", True):
                    try:
                        prod_model_record = await get_production_model(async_session, self.study_name)
                        if prod_model_record and prod_model_record.model_artifact_url:
                            logger.info("warm_start_model_identified", model_id=str(prod_model_record.id))
                    except Exception as e:
                        logger.warning("failed_to_load_base_model", error=str(e))

            x_vals, y_vals, feature_names, dataset_metadata = self._prepare_training_data(df_featured)
            study = await self._train_and_optimize(x_vals, y_vals, feature_names, dataset_metadata, base_model)
            best_accuracy = study.best_value
            self._export_model(best_accuracy)

            is_drifted = self.performance_monitor.detect_drift(best_accuracy)
            self.performance_monitor.add_metric(best_accuracy)
            logger.info("pipeline_completed", best_accuracy=best_accuracy, performance_drift=is_drifted, best_params=study.best_params)
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
        "framework": os.getenv("FRAMEWORK", "xgboost")
    }
    pipeline = AutonomousMLPipeline(config)
    asyncio.run(pipeline.run())
