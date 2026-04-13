from collections import deque
from typing import Any, cast

import numpy as np
import structlog
from scipy.stats import ks_2samp

from src.shared.observability import PERFORMANCE_DRIFT_ALERT

logger = structlog.get_logger(__name__)


class PerformanceDriftMonitor:
    """
    Monitors model performance metrics for degradation over time.
    Uses a rolling window of historical values as a baseline.
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 0.05,
        higher_is_better: bool = True,
        model_name: str = "default",
    ) -> None:
        self.history: deque[float] = deque(maxlen=window_size)
        self.threshold = threshold
        self.window_size = window_size
        self.higher_is_better = higher_is_better
        self.model_name = model_name
        self.redis_key = f"ml:perf_history:{model_name}"
        self._sync_with_redis()

    def _sync_with_redis(self) -> None:
        """Sync local history with Redis state (Synchronous attempt)."""
        import asyncio

        from src.shared.utils.cache import get_redis

        redis = get_redis()
        if not redis:
            return

        async def _load() -> None:
            try:
                data = await redis.get(self.redis_key)
                if data:
                    import msgspec

                    metrics = cast(list[float], msgspec.json.decode(data))
                    self.history.clear()
                    self.history.extend(metrics)
                    logger.info(
                        "metrics_baseline_recovered", model=self.model_name, count=len(metrics)
                    )
            except Exception as e:
                logger.warning("metrics_recovery_failed", error=str(e))

        try:
            # We try to run this if we are not already in a loop,
            # or just fire and forget if we are.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_load())
            else:
                loop.run_until_complete(_load())
        except Exception:
            pass

    def add_metric(self, value: float) -> None:
        """Adds a new performance metric to the historical baseline and persists to Redis."""
        self.history.append(value)

        from src.shared.utils.cache import get_redis

        redis = get_redis()
        if redis:

            async def persist() -> None:
                try:
                    import msgspec

                    await redis.set(self.redis_key, msgspec.json.encode(list(self.history)))
                except Exception as e:
                    logger.warning("metrics_persistence_failed", error=str(e))

            try:
                import asyncio

                loop = asyncio.get_event_loop()
                loop.create_task(persist())
            except Exception:
                pass

    def detect_drift(self, current_value: float) -> bool:
        """Detects performance degradation compared to historical baseline."""
        if len(self.history) < self.window_size:
            return False

        baseline = sum(self.history) / len(self.history)
        if self.higher_is_better:
            is_drifted = current_value < (baseline - self.threshold)
        else:
            is_drifted = current_value > (baseline + self.threshold)

        PERFORMANCE_DRIFT_ALERT.set(1 if is_drifted else 0)
        return bool(is_drifted)


def calculate_ks_test(
    expected: np.ndarray[Any, np.dtype[np.float64]],
    actual: np.ndarray[Any, np.dtype[np.float64]] | list[float],
) -> tuple[float, float]:
    """
    Performs Kolmogorov-Smirnov test to detect data drift.
    Returns (statistic, p_value).
    """
    res = ks_2samp(expected, actual)
    return float(res.statistic), float(res.pvalue)


def calculate_psi(
    expected: np.ndarray[Any, np.dtype[np.float64]],
    actual: np.ndarray[Any, np.dtype[np.float64]],
    buckets: int = 10,
) -> float:
    """
    Calculates Population Stability Index (PSI).
    """

    def scale_range(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))

    if max_val == min_val:
        return 0.0

    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)

    expected_percents = np.histogram(expected_scaled, bins=buckets, range=(0, 1))[0] / len(expected)
    actual_percents = np.histogram(actual_scaled, bins=buckets, range=(0, 1))[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)

    psi_value = np.sum(
        (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    )
    return float(psi_value)


class DriftTrigger:
    """
    God-Mode: Automated Retraining Trigger.
    Combines Statistical Drift (PSI/KS) with Performance Degradation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.psi_threshold = config.get("psi_threshold", 0.2)
        self.perf_threshold = config.get("perf_threshold", 0.05)

    def should_retrain(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        current_performance: float,
        baseline_performance: float | None = None,
    ) -> tuple[bool, str]:
        """
        Determines if retraining is necessary based on data drift and performance.
        """
        if self.config.get("force_train", False):
            return True, "force_train_enabled"

        # 1. Statistical Data Drift (PSI)
        psi = calculate_psi(reference_data, current_data)
        if psi > self.psi_threshold:
            return True, f"data_drift_detected_psi_{psi:.4f}"

        # 2. Performance Drift
        if baseline_performance is not None:
            if current_performance < (baseline_performance - self.perf_threshold):
                return True, f"perf_drift_detected_{current_performance:.4f}"

        return False, "no_drift_detected"