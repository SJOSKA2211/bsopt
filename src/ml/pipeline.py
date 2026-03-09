import asyncio
from typing import Any

import numpy as np
import pandas as pd
import structlog
from numba import njit, prange

logger = structlog.get_logger(__name__)


class PipelineConfig:
    """Configuration for the data pipeline."""

    def __init__(
        self,
        symbols: list[str] = ["AAPL"],
        min_samples: int = 1000,
        max_samples: int = 10000,
        use_multi_source: bool = False,
        validate_data: bool = True,
        storage_backend: str = "database",
        output_dir: str = "data/training",
    ):
        self.symbols = symbols
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.use_multi_source = use_multi_source
        self.validate_data = validate_data
        self.storage_backend = storage_backend
        self.output_dir = output_dir


class DataPipeline:
    """
    Data Pipeline for collecting and processing market data.
    OPTIMIZED: Loads real data from Postgres.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_run_report: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """
        Run the data collection pipeline.
        """
        logger.info("data_pipeline_start", symbols=self.config.symbols)

        from src.database.pipeliner import db_engine

        data = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        self.last_run_report = {
            "samples_available": len(data),
            "symbols": self.config.symbols,
            "duration_seconds": 1.0,
            "status": "ready",
        }
        logger.info("data_pipeline_status", report=self.last_run_report)
        return self.last_run_report

    async def load_latest_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
        """
        Load the latest collected data from Postgres (Optimized extraction).
        Returns: (X, y, feature_names, metadata)
        """
        from src.database.pipeliner import db_engine

        records = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        if not records:
            from src.ml.training.data_gen import (
                generate_synthetic_data_numba as generate_synthetic_data,
            )

            logger.warning("data_pipeline_no_real_data_found", fallback="synthetic")
            X, y, features = generate_synthetic_data(self.config.min_samples)
            return X, y, features, {"data_source": "synthetic_numba", "count": len(X)}

        # OPTIMIZED: Vectorized extraction using NumPy from record list
        strikes = np.array([r["strike"] for r in records], dtype=np.float64)
        last_prices = np.array([r["last"] for r in records], dtype=np.float64)
        ivs = np.array([r["implied_volatility"] or 0.2 for r in records], dtype=np.float64)

        # Handle datetimes efficiently
        expiries = np.array(
            [r["expiry"].timestamp() if hasattr(r["expiry"], "timestamp") else 0.0 for r in records],
            dtype=np.float64
        )
        times = np.array(
            [r["time"].timestamp() if hasattr(r["time"], "timestamp") else 0.0 for r in records],
            dtype=np.float64
        )

        maturities = _calculate_maturity_jit(expiries, times)
        maturities = np.where(maturities <= 0, 0.5, maturities)

        n = len(records)
        X_base = np.empty((n, 5), dtype=np.float64)
        X_base[:, 0] = strikes
        X_base[:, 1] = maturities
        X_base[:, 2] = ivs
        X_base[:, 3] = 0.05  # Rate
        X_base[:, 4] = 0.01  # Dividend
        
        y_raw = last_prices

        iv_lag = np.roll(ivs, 1)
        iv_lag[0] = ivs[0]
        price_lag = np.roll(y_raw, 1)
        price_lag[0] = y_raw[0]

        iv_ma5 = _rolling_mean_jit(ivs, 5)
        iv_ma20 = _rolling_mean_jit(ivs, 20)
        price_ma5 = _rolling_mean_jit(y_raw, 5)
        price_ma20 = _rolling_mean_jit(y_raw, 20)

        X = np.column_stack([X_base, iv_lag, price_lag, iv_ma5, iv_ma20, price_ma5, price_ma20])

        feature_names = ["strike", "maturity", "iv", "rate", "dividend"]
        feature_names += [
            "iv_lag1",
            "price_lag1",
            "iv_ma5",
            "iv_ma20",
            "price_ma5",
            "price_ma20",
        ]
        y = y_raw

        metadata = {"data_source": "postgres_jit_vectorized", "count": n}
        return X, y, feature_names, metadata


class MLPipeline:
    """
    Unified Autonomous ML Pipeline.
    Wires together Data Collection, Training, and Evaluation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.symbols = [config.get("ticker", "AAPL")]
        self.data_pipeline = DataPipeline(PipelineConfig(symbols=self.symbols))
        from src.ml.trainer import ModelTrainer

        self.trainer = ModelTrainer(
            study_name=config.get("study_name", "autonomous_pipeline"),
            tracking_uri=config.get("tracking_uri"),
        )

    async def run(self, force: bool = False) -> Any:
        """
        Executes the full pipeline: Data -> Train -> Model.
        """
        logger.info("ml_pipeline_run_start", ticker=self.symbols[0])

        # 1. Fetch Data
        await self.data_pipeline.run()
        X, y, features, meta = await self.data_pipeline.load_latest_data()

        # 2. Train and Evaluate
        params = self.config.copy()
        # Default to XGBoost if not specified
        params["framework"] = self.config.get("framework", "xgboost")

        # Run training (Synchronous in ModelTrainer, but we could wrap in to_thread)
        score = await asyncio.to_thread(
            self.trainer.train_and_evaluate, X, y, params, features, meta
        )

        logger.info("ml_pipeline_run_complete", score=score)

        # 3. Model Registration and Promotion
        if self.trainer.model:
            from src.ml.registry.promote import promote_model

            run_id = self.trainer.tracker.current_run.info.run_id
            model_name = f"{params['framework']}_pricer_{self.symbols[0]}"
            logger.info("promoting_new_champion", model=model_name, run_id=run_id)

            # In a real scenario, we would compare scores before promoting
            # For now, we promote the latest successful run
            promote_model(model_name, run_id, stage="Production")

        return self.trainer.model

    async def shutdown(self):
        """Cleanup resources."""
        pass


if __name__ == "__main__":
    import argparse

    import mlflow

    parser = argparse.ArgumentParser(description="BS-OPT Autonomous ML Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--framework", type=str, default="xgboost")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="regressor_v1")
    parser.add_argument("--tracking_uri", type=str, default=None)

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    async def main():
        config = {
            "ticker": args.ticker,
            "framework": args.framework,
            "n_trials": args.n_trials,
            "study_name": args.study_name,
            "tracking_uri": args.tracking_uri,
        }
        pipeline = MLPipeline(config)
        await pipeline.run()
        await pipeline.shutdown()

    asyncio.run(main())


# =============================================================================
# Helper Functions (Numba JIT)
# =============================================================================


@njit(fastmath=True, parallel=True)
def _rolling_mean_jit(x: np.ndarray, w: int) -> np.ndarray:
    """Numba-optimized rolling mean with same-length padding (float64)."""
    n = len(x)
    res = np.empty(n, dtype=np.float64)
    if n < w:
        res[:] = x[0]
        return res

    # Initial padding
    res[: w - 1] = x[0]

    # Calculate initial sum
    current_sum = 0.0
    for i in prange(w):
        current_sum += x[i]
    res[w - 1] = current_sum / w

    # Sliding window
    for i in prange(w, n):
        current_sum += x[i] - x[i - w]
        res[i] = current_sum / w
    return res


@njit(fastmath=True, parallel=True)
def _calculate_maturity_jit(
    expiry_timestamps: np.ndarray, current_timestamps: np.ndarray
) -> np.ndarray:
    """Vectorized maturity calculation."""
    return (expiry_timestamps - current_timestamps) / (365.0 * 24 * 3600)


# =============================================================================
# Data Pipeline Implementation
# =============================================================================


def _check_cache(file_path: str) -> bool:
    """
    Simulates checking a cache. In a real scenario, this would interact with Redis or a file system.
    Returns True if cache is considered valid, False otherwise.
    """
    # This is a placeholder. In a real system, you'd check file modification time,
    # or a cache entry's expiry in Redis.
    # For now, assume cache is always stale for demonstration purposes.
    return False


async def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Real feature computation using the centralized Feature Store (Optimized)."""
    from src.ml.feature_store.store import feature_store
    
    logger.info("computing_features_production", count=len(df))
    # Required features for pricing/training
    required = ["log_return", "EMA_20", "RSI_14", "MACD"]
    
    try:
        # feature_store handles Numba-accelerated computation internally
        return await feature_store.compute_features(df, required)
    except Exception as e:
        logger.error("feature_computation_failed", error=str(e))
        return df


async def _background_cache_fill(df: pd.DataFrame):
    """Placeholder for background cache population."""
    logger.info("background_cache_fill_simulated")
    # Simulate writing to cache
    await asyncio.sleep(0.01)
    pass
