import asyncio

import msgspec
import numpy as np
import pandas as pd
import structlog
from numba import njit

from src.config import settings
from src.ml.training.base import TrainingConfig, TrainingResult

logger = structlog.get_logger(__name__)

class PipelineConfig(msgspec.Struct):
    """Configuration for the data pipeline."""
    symbols: list[str] = [settings.DEFAULT_TICKER]
    min_samples: int = 1000
    max_samples: int = 10000
    validate_data: bool = True
    output_dir: str = "data/training"

class DataPipeline:
    """
    Data Pipeline for collecting and processing market data.
    OPTIMIZED: Loads real data from Postgres.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_run_report: dict[str, str | int | float] = {}

    async def run(self) -> None:
        """
        Run the data collection pipeline.
        """
        logger.info("data_pipeline_start", symbols=self.config.symbols)

        from src.database.pipeliner import db_engine

        data = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        self.last_run_report = {
            "samples_available": len(data),
            "symbols": ",".join(self.config.symbols),
            "status": "ready",
        }

    async def run_shutdown(self) -> None:
        """Cleanup data pipeline resources."""
        logger.info("data_pipeline_shutdown_complete")

    async def load_latest_data(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[str],
        dict[str, str | int],
    ]:
        """
        Load the latest collected data from Postgres (Optimized cross-sectional extraction).
        Returns: (X, y, feature_names, metadata)
        """
        import pandas as pd

        from src.database.pipeliner import db_engine
        from src.shared.math_utils import calculate_greeks

        # Use chunked extraction to handle large cross-sectional datasets efficiently
        chunk_size = 50000
        records = []
        offset = 0
        while True:
            chunk = await db_engine.fetch_training_data(
                self.config.symbols, limit=chunk_size, offset=offset
            )
            if not chunk:
                break
            records.extend(chunk)
            offset += chunk_size
            if len(records) >= self.config.max_samples:
                break

        if not records:
            logger.error("data_pipeline_no_real_data_found", symbols=self.config.symbols)
            raise ValueError(f"No real market data found in Postgres for symbols: {self.config.symbols}")

        # Load into Pandas for vectorized cleaning and robust cross-sectional manipulation
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by=["symbol", "time"])

        # Rigorous Data Cleaning: handle NaNs, forward-fills
        from src.ml.pre_training import MLPreTrainer

        # Use MLPreTrainer for cross-sectional features and advanced imputation
        df = MLPreTrainer.calculate_cross_sectional_features(df)

        # Feature names for imputation
        base_features = ["last", "strike", "implied_volatility"]
        if "option_type" in df.columns:
            df["is_call"] = (df["option_type"] == "call").astype(float)
            base_features.append("is_call")

        # Perform advanced imputation (Spline + Forward Fill)
        # Note: We do this per symbol to maintain time-series integrity
        processed_dfs = []
        for sym, group in df.groupby("symbol"):
            normalized_data, _, _ = MLPreTrainer.prepare_features(group, base_features)
            # Re-assign imputed values back to dataframe
            for idx, feat in enumerate(base_features):
                group[feat] = normalized_data[:, idx]
            processed_dfs.append(group)

        df = pd.concat(processed_dfs)

        # Feature Engineering: Compute base vectors
        s = df["last"].values
        k = df["strike"].values
        t = np.where(
            _calculate_maturity_jit(
                pd.to_datetime(df["expiry"]).astype("int64").values // 10**9,
                df["time"].astype("int64").values // 10**9,
            )
            <= 0,
            0.5,
            _calculate_maturity_jit(
                pd.to_datetime(df["expiry"]).astype("int64").values // 10**9,
                df["time"].astype("int64").values // 10**9,
            ),
        )

        sigma = df["implied_volatility"].fillna(settings.DEFAULT_VOLATILITY).values
        r = np.full_like(s, settings.RISK_FREE_RATE)
        q = np.full_like(s, settings.DIVIDEND_YIELD)
        is_call = (df.get("option_type", "call") == "call").values

        # Calculate cross-sectional Black-Scholes Features
        delta, gamma, theta, vega, rho = calculate_greeks(s, k, t, sigma, r, q, is_call)
        df["delta"] = delta
        df["gamma"] = gamma
        df["vega"] = vega

        # Cross-sectional Targets: Predict next price conditionally
        df["target_price"] = df.groupby("symbol")["last"].shift(-1).fillna(df["last"])

        X = df[["strike", "delta", "gamma", "vega", "implied_volatility"]].values
        y = df["target_price"].values
        feature_names = ["strike", "delta", "gamma", "vega", "iv"]
        metadata = {
            "data_source": "postgres_chunked_pandas",
            "count": len(X),
        }

        return X, y, feature_names, metadata

class MLPipeline:
    """
    Unified Autonomous ML Pipeline.
    Wires together Data Collection, Training, and Evaluation.
    """

    def __init__(self, training_config: TrainingConfig, pipeline_config: PipelineConfig | None = None):
        self.training_config = training_config
        self.pipeline_config = pipeline_config or PipelineConfig(symbols=training_config.metadata.get("ticker", [settings.DEFAULT_TICKER]).split(","))
        self.data_pipeline = DataPipeline(self.pipeline_config)
        
        from src.ml.trainer import ModelTrainer
        self.trainer = ModelTrainer(
            study_name=training_config.metadata.get("study_name", "autonomous_pipeline")
        )

    async def run(self) -> TrainingResult:
        """
        Executes the full pipeline: Data -> Train -> Model.
        """
        logger.info("ml_pipeline_run_start", symbols=self.pipeline_config.symbols)

        # 1. Fetch Data
        await self.data_pipeline.run()
        X, y, features, meta = await self.data_pipeline.load_latest_data()

        # 2. Train and Evaluate
        # Run training strictly typed
        result = await asyncio.to_thread(
            self.trainer.train_and_evaluate, X, y, self.training_config, meta
        )

        logger.info("ml_pipeline_run_complete", score=result.score)

        # 3. Model Registration and Promotion
        if self.trainer.model:
            from src.ml.registry.promote import promote_model
            
            # Use active run from MLflow
            run = mlflow.active_run()
            if run:
                run_id = run.info.run_id
                model_name = f"{self.training_config.framework}_pricer_{self.pipeline_config.symbols[0]}"
                logger.info("promoting_new_champion", model=model_name, run_id=run_id)
                promote_model(model_name, run_id, stage="Production")

        return result

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("ml_pipeline_shutdown")
        await self.data_pipeline.run_shutdown()
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    import argparse

    import mlflow

    parser = argparse.ArgumentParser(description="BS-OPT Autonomous ML Pipeline")
    parser.add_argument("--ticker", type=str, default=settings.DEFAULT_TICKER)
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
def _calculate_maturity_jit(
    expiry_timestamps: np.ndarray, current_timestamps: np.ndarray
) -> np.ndarray:
    """Vectorized maturity calculation."""
    return (expiry_timestamps - current_timestamps) / (365.0 * 24 * 3600)

# =============================================================================
# Data Pipeline Implementation
# =============================================================================

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
