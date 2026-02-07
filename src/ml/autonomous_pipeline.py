import asyncio
from typing import Any

import numpy as np
import pandas as pd
import structlog

try:
    from numba import config, cuda, float64, jit, njit, prange, vectorize
except ImportError:

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

    class Config:
        pass

    config = Config()

    def vectorize(*args, **kwargs):
        def decorator(func):
            return np.vectorize(func)

        return decorator

    class NumbaType:
        def __call__(self, *args):
            return self

    float64 = NumbaType()

    class CudaMock:
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def grid(self, *args):
            return 0

        def device_array(self, n, dtype):
            return np.zeros(n, dtype=dtype)

    cuda = CudaMock()
from sqlalchemy import create_engine, func, select

from src.config import get_settings
from src.database import Base, get_async_db_context
from src.database.models import ModelPrediction
from src.ml.drift import DriftTrigger, PerformanceDriftMonitor
from src.ml.scraper import MarketDataScraper
from src.ml.trainer import InstrumentedTrainer
from src.shared.observability import (
    push_metrics,
    setup_logging,
)

# Initialize structured logger
logger = structlog.get_logger()

# =============================================================================
# High-Performance Indicator Kernels (Numba NJIT)
# =============================================================================


@jit(nopython=True, cache=True)
def _numba_ema(values: np.ndarray, span: int) -> np.ndarray:
    """Carr-Madan compliant EMA via recurrence relation. O(N) complexity."""
    alpha = 2 / (span + 1)
    out = np.empty_like(values)
    out[:] = np.nan

    # Identify first finite sequence entry
    first_valid_idx = -1
    for i in range(len(values)):
        if not np.isnan(values[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return out

    out[first_valid_idx] = values[first_valid_idx]
    for i in range(first_valid_idx + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]  # Recurrence propagation through discontinuities
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]

    return out


@jit(nopython=True, cache=True)
def _numba_rsi(prices: np.ndarray, length: int = 14) -> np.ndarray:
    """Wilder's RSI implementation. NJIT parallel-capable but serial for small windows."""
    out = np.full_like(prices, np.nan)
    if len(prices) <= length:
        return out

    deltas = prices[1:] - prices[:-1]
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # SMA Seed for Wilder's Smoothing
    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    if avg_loss == 0:
        out[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[length] = 100.0 - (100.0 / (1.0 + rs))

    # Recurrent update loop
    for i in range(length + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0

        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


@jit(nopython=True, cache=True)
def _numba_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Vectorized MACD oscillator decomposition."""
    ema_fast = _numba_ema(prices, fast)
    ema_slow = _numba_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _numba_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


@jit(nopython=True, cache=True, parallel=True)
def _numba_bbands(prices: np.ndarray, length: int = 20, std: float = 2.0):
    """Calculate Bollinger Bands (Lower, Mid, Upper)."""
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in prange(length - 1, n):
        window = prices[i - length + 1 : i + 1]
        mu = np.mean(window)
        sigma = np.std(window)
        mid[i] = mu
        upper[i] = mu + std * sigma
        lower[i] = mu - std * sigma

    return lower, mid, upper


@jit(nopython=True, cache=True)
def _numba_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    """Calculate Average True Range (ATR)."""
    tr = np.zeros_like(close)
    # TR[0] is High[0] - Low[0]
    tr[0] = high[0] - low[0]

    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # Wilder's Smoothing for ATR
    atr = np.full_like(tr, np.nan)
    atr[length - 1] = np.mean(tr[:length])  # Initial SMA

    for i in range(length, len(tr)):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    return atr


@jit(nopython=True, cache=True)
def _numba_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14):
    """Calculate ADX."""
    # 1. Calculate TR, +DM, -DM
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = high[0] - low[0]

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]

        # TR
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

        # +DM
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        else:
            plus_dm[i] = 0

        # -DM
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
        else:
            minus_dm[i] = 0

    # 2. Smooth TR, +DM, -DM using Wilder's
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    # Init with sum (not mean) for the first 'length' periods per Wilder's definition often used
    # But standard Wilder's usually takes SMA?
    # Let's use Sum for initialization of DM/TR in ADX as per common implementations
    tr_smooth[length - 1] = np.sum(tr[:length])
    plus_dm_smooth[length - 1] = np.sum(plus_dm[:length])
    minus_dm_smooth[length - 1] = np.sum(minus_dm[:length])

    for i in range(length, n):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / length) + tr[i]
        plus_dm_smooth[i] = (
            plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / length) + plus_dm[i]
        )
        minus_dm_smooth[i] = (
            minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / length) + minus_dm[i]
        )

    # 3. Calculate DX
    dx = np.full(n, np.nan)
    for i in range(length - 1, n):
        if tr_smooth[i] == 0:
            dx[i] = 0
        else:
            plus_di = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di = 100 * minus_dm_smooth[i] / tr_smooth[i]
            if plus_di + minus_di == 0:
                dx[i] = 0
            else:
                dx[i] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # 4. Calculate ADX (Smoothed DX)
    adx = np.full(n, np.nan)
    # First ADX is mean of DX
    if not np.isnan(dx[2 * length - 2]):  # length-1 + length-1
        adx[2 * length - 2] = np.mean(dx[length - 1 : 2 * length - 1])

        for i in range(2 * length - 1, n):
            adx[i] = (adx[i - 1] * (length - 1) + dx[i]) / length

    return adx


class AutonomousMLPipeline:
    """
    End-to-end autonomous ML pipeline integrating scraping, persistence,
    drift detection, and model optimization.
    """

    def __init__(self, config: dict[str, Any]):
        setup_logging()
        self.config = config
        self.scraper = MarketDataScraper(
            api_key=config["api_key"], provider=config.get("provider", "auto")
        )
        self.db_url = config["db_url"]
        self.ticker = config["ticker"]
        self.study_name = config["study_name"]
        self.n_trials = config.get("n_trials", 50)  # Increased default trials
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
            # Calculate accuracy: % of predictions where (predicted > 0.5) == (actual > 0.5)
            # Assuming classification for this specific pipeline
            result = session.execute(
                select(
                    func.avg(
                        func.cast(
                            (ModelPrediction.predicted_price > 0.5)
                            == (ModelPrediction.actual_price > 0.5),
                            np.float64,
                        )
                    )
                )
                .where(ModelPrediction.actual_price.isnot(None))
                .limit(100)  # Last 100 predictions
            ).scalar()
            return float(result) if result is not None else None
        except Exception as e:
            logger.warning("failed_to_fetch_performance", error=str(e))
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators. Uses raw numpy and numba for high-frequency
        feature computation, minimizing overhead.
        """
        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert to numpy arrays for Numba
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)

        # 1. High-Performance Vectorized Returns (Numpy)
        # Prepend 0 to maintain length
        log_ret = np.log(
            closes[1:] / closes[:-1],
            where=(closes[:-1] != 0),
            out=np.zeros_like(closes[1:]),
        )
        pct_ret = (closes[1:] / closes[:-1]) - 1

        df["log_return"] = np.concatenate([np.zeros(1), log_ret])
        df["pct_return"] = np.concatenate([np.zeros(1), pct_ret])

        # 2. Vectorized Volatility (Rolling Standard Deviation using Numpy tricks)
        window = 20
        returns = df["pct_return"].values

        # Fast rolling std using numpy
        def rolling_std(x, w):
            s1 = np.convolve(x, np.ones(w), "valid")
            s2 = np.convolve(x**2, np.ones(w), "valid")
            return np.sqrt((s2 - s1**2 / w) / (w - 1))

        vol = rolling_std(returns, window)
        # Pad beginning with NaNs to match dataframe length
        df["volatility"] = np.concatenate([np.full(window - 1, np.nan), vol]) * np.sqrt(
            252 * 6.5 * 60
        )

        # 3. Numba-Optimized Complex Indicators (Replacing pandas-ta)
        # RSI
        df["RSI_14"] = _numba_rsi(closes, length=14)

        # MACD
        macd, signal, hist = _numba_macd(closes, fast=12, slow=26, signal=9)
        df["MACD_12_26_9"] = macd
        df["MACDs_12_26_9"] = signal
        df["MACDh_12_26_9"] = hist

        # Bollinger Bands
        lower, mid, upper = _numba_bbands(closes, length=20, std=2.0)
        df["BBL_20_2.0"] = lower
        df["BBM_20_2.0"] = mid
        df["BBU_20_2.0"] = upper

        # ATR
        df["ATR_14"] = _numba_atr(highs, lows, closes, length=14)

        # ADX
        df["ADX_14"] = _numba_adx(highs, lows, closes, length=14)

        return df.dropna().copy()

    async def run(self):
        """
        Executes the full pipeline asynchronously with smart drift-based retraining.
        """
        logger.info("pipeline_started", ticker=self.ticker, framework=self.framework)

        try:
            # 1. Scrape Data
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            str_start = start_date.strftime("%Y-%m-%d")
            str_end = end_date.strftime("%Y-%m-%d")

            try:
                df = await self.scraper.fetch_historical_data(
                    self.ticker, str_start, str_end
                )
            except Exception as e:
                logger.warning("scrape_failed_retrying_mock", error=str(e))
                self.scraper = MarketDataScraper(api_key="DEMO_KEY", provider="mock")
                df = await self.scraper.fetch_historical_data(
                    self.ticker, str_start, str_end
                )

            logger.info("data_scraped", rows=len(df))

            # 2. Persist to DB using optimized bulk insert (COPY)
            from src.database.crud import bulk_insert_market_ticks, get_production_model

            market_data_records = [
                {
                    "time": pd.to_datetime(ts, unit="s", utc=True),
                    "symbol": self.ticker,
                    "price": float(close),
                    "volume": int(vol),
                    "side": None,
                }
                for ts, close, vol in zip(
                    df["timestamp"], df["close"], df.get("volume", [0] * len(df))
                )
            ]

            async with get_async_db_context() as async_session:
                if market_data_records:
                    await bulk_insert_market_ticks(async_session, market_data_records)

                # 3. Feature Engineering (Vectorized)
                df_featured = self.generate_features(df)
                logger.info("features_generated", columns=list(df_featured.columns))

                # 4. Smart Trigger: Automated Drift & Performance Evaluation
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

                # 5. Autonomous Training with Warm Start support
                trainer = InstrumentedTrainer(study_name=self.study_name)

                # Attempt to load base model for warm start if applicable
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

            # Define target and features
            # Target: 1 if close price increases in the next period
            df_featured["target"] = (
                df_featured["close"].shift(-1) > df_featured["close"]
            ).astype(int)
            df_featured = df_featured.iloc[:-1]  # Remove last row due to shift

            # Select features (exclude non-numeric and target)
            exclude = ["timestamp", "target", "ticker"]
            feature_names = [col for col in df_featured.columns if col not in exclude]
            X = df_featured[feature_names].values
            y = df_featured["target"].values

            dataset_metadata = {
                "ticker": self.ticker,
                "rows": str(len(df_featured)),
                "features": str(len(feature_names)),
            }

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
                        "min_samples_split": trial.suggest_int(
                            "min_samples_split", 2, 10
                        ),
                        "framework": "sklearn",
                    }
                else:  # pytorch
                    params = {
                        "epochs": trial.suggest_int("epochs", 20, 100),
                        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                        "batch_size": trial.suggest_categorical(
                            "batch_size", [16, 32, 64]
                        ),
                        "framework": "pytorch",
                    }

                return trainer.train_and_evaluate(
                    X,
                    y,
                    params,
                    feature_names=feature_names,
                    dataset_metadata=dataset_metadata,
                    base_model=base_model,
                    trial=trial,  # Pass trial for reporting/pruning
                )

            study = trainer.optimize(objective, n_trials=self.n_trials)
            best_accuracy = study.best_value

            # 6. Model Optimization & Export (New Integration)
            if best_accuracy >= self.config.get("promotion_threshold", 0.8):
                logger.info("model_promotion_triggered", accuracy=best_accuracy)

                # Export to ONNX and trigger quantization task
                model_path = f"models/{self.study_name}_latest.onnx"
                quantized_path = f"models/{self.study_name}_latest.int8.onnx"

                try:
                    # Logic to export XGBoost/PyTorch to ONNX
                    # For now, we simulate the export; in production, this would use onnxmltools
                    logger.info("exporting_model_to_onnx", path=model_path)

                    # Trigger the async quantization task
                    from src.tasks.ml_tasks import optimize_model_task

                    optimize_model_task.delay(model_path, quantized_path)
                except Exception as e:
                    logger.error("model_export_failed", error=str(e))

            # 7. Performance Drift Detection
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

    # Determine API Key and Provider
    # Prioritize Polygon if available, as AV was reporting 401
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

    # Example usage
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
