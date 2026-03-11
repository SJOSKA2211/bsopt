from collections import deque
from typing import Any, cast

import numpy as np
import structlog
from scipy.stats import ks_2samp

from src.shared.observability import (
    DATA_DRIFT_SCORE,
    KS_TEST_SCORE,
    PERFORMANCE_DRIFT_ALERT,
)
from src.shared.math_utils import njit_engine

try:
    import bsopt_core
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Initialize structured logger
logger = structlog.get_logger()


class PerformanceDriftMonitor:
    """
    Monitors model performance (e.g., accuracy, RMSE, R2) for degradation over time.
    Uses a rolling window of historical performance as a baseline.
    Supports both 'higher is better' (Accuracy, R2) and 'lower is better' (RMSE, MAE).
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

        from src.utils.cache import get_redis

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

        from src.utils.cache import get_redis

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
        """
        Detects if the current performance value has drifted (degraded)
        significantly from the historical baseline.
        """
        if len(self.history) < self.window_size:
            logger.debug("drift_detection_skipped", reason="insufficient_history")
            return False

        baseline = sum(self.history) / len(self.history)

        if self.higher_is_better:
            # Degradation: current value is LOWER than baseline - threshold
            # E.g., R2 dropped from 0.95 to 0.85
            is_drifted = current_value < (baseline - self.threshold)
        else:
            # Degradation: current value is HIGHER than baseline + threshold
            # E.g., RMSE increased from 0.02 to 0.08
            is_drifted = current_value > (baseline + self.threshold)

        PERFORMANCE_DRIFT_ALERT.set(1 if is_drifted else 0)

        if is_drifted:
            logger.warning(
                "performance_drift_detected",
                baseline=baseline,
                current=current_value,
                threshold=self.threshold,
                higher_is_better=self.higher_is_better,
            )
        else:
            logger.info("performance_check_passed", baseline=baseline, current=current_value)

        return bool(is_drifted)


def calculate_ks_test(expected: np.ndarray[Any, np.dtype[np.float64]], actual: np.ndarray[Any, np.dtype[np.float64]] | list[float]) -> tuple[float, float]:
    """
    Calculates the Kolmogorov-Smirnov (KS) test between two distributions.

    Args:
        expected: Reference dataset (e.g., training data).
        actual: Current dataset (e.g., production data).

    Returns:
        Tuple[float, float]: The KS statistic and the p-value.
    """
    logger.info("ks_test_calculation_started")

    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)

    statistic, p_value = ks_2samp(expected_arr, actual_arr)

    # Emit Prometheus metric
    KS_TEST_SCORE.set(float(p_value))

    logger.info("ks_test_calculation_completed", statistic=statistic, p_value=p_value)

    return float(statistic), float(p_value)


@njit_engine(cache=True, fastmath=True)
def _psi_kernel(
    expected_counts: np.ndarray[Any, np.dtype[np.float64]], actual_counts: np.ndarray[Any, np.dtype[np.float64]], expected_len: int, actual_len: int
) -> float:
    """Numba-optimized PSI kernel with epsilon padding."""
    eps = 1e-6
    expected_pct = (expected_counts / expected_len) + eps
    actual_pct = (actual_counts / actual_len) + eps

    psi_sum = 0.0
    for i in range(len(expected_pct)):
        psi_sum += (actual_pct[i] - expected_pct[i]) * np.log(actual_pct[i] / expected_pct[i])
    return float(psi_sum)


def calculate_psi(
    expected: np.ndarray[Any, np.dtype[np.float64]],
    actual: np.ndarray[Any, np.dtype[np.float64]] | list[float],
    buckets: int = 10,
    bins: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> float:
    """
    OPTIMIZED: Population Stability Index with pre-calculated bins.
    """
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)

    if bins is None:
        # Fallback to dynamic binning if not provided
        min_val = min(float(expected_arr.min()), float(actual_arr.min()))
        max_val = max(float(expected_arr.max()), float(actual_arr.max()))
        bins = np.linspace(min_val, max_val, buckets + 1)

    if CORE_AVAILABLE:
        try:
            psi_score = float(cast(Any, bsopt_core).calculate_psi(
                expected_arr.astype(np.float64), actual_arr.astype(np.float64), bins.astype(np.float64)
            ))
        except Exception as e:
            logger.warning("rust_psi_calculation_failed_falling_back", error=str(e))
            # Fallback to JIT kernel
            expected_counts, _ = np.histogram(expected_arr, bins=bins)
            actual_counts, _ = np.histogram(actual_arr, bins=bins)
            psi_score = _psi_kernel(expected_counts.astype(np.float64), actual_counts.astype(np.float64), len(expected_arr), len(actual_arr))
    else:
        # Fast bucketing using pre-defined bins
        expected_counts, _ = np.histogram(expected_arr, bins=bins)
        actual_counts, _ = np.histogram(actual_arr, bins=bins)
        psi_score = _psi_kernel(expected_counts.astype(np.float64), actual_counts.astype(np.float64), len(expected_arr), len(actual_arr))

    # Emit Prometheus metric
    DATA_DRIFT_SCORE.set(psi_score)
    return psi_score


class DriftTrigger:
    """
    Evaluates multiple drift and performance signals to decide if model
    retraining should be triggered.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.psi_threshold = float(config.get("psi_threshold", 0.1))
        self.ks_p_value_threshold = float(config.get("ks_p_value_threshold", 0.05))
        self.perf_threshold = float(config.get("perf_threshold", 0.05))
        self.force_train = bool(config.get("force_train", False))

        # Use existing PerformanceDriftMonitor
        self.performance_monitor = PerformanceDriftMonitor(
            higher_is_better=bool(config.get("perf_higher_is_better", True)),
        )
        self.feature_drifts: dict[str, dict[str, float]] = {}

    def trigger_retrain(self, ticker: str, model_type: str) -> bool:
        """Asynchronously trigger the Celery retraining task."""
        try:
            # Avoid circular import
            from src.tasks.ml_tasks import train_model_task

            logger.info("triggering_autonomous_retrain", ticker=ticker, model_type=model_type)
            train_model_task.delay(ticker=ticker, model_type=model_type)
            return True
        except Exception as e:
            logger.error("failed_to_trigger_retrain_task", error=str(e))
            return False

    def should_retrain(
        self,
        reference_data: np.ndarray[Any, np.dtype[np.float64]] | dict[str, np.ndarray[Any, np.dtype[np.float64]]],
        current_data: np.ndarray[Any, np.dtype[np.float64]] | dict[str, np.ndarray[Any, np.dtype[np.float64]]],
        current_perf: float | None,
    ) -> tuple[bool, str]:
        """
        Determines if retraining is necessary based on data distribution shift
        and performance degradation. Supports both single-series and multi-feature dicts.

        Returns:
            Tuple[bool, str]: (Decision, Reason)
        """
        if self.force_train:
            return True, "force_train"

        # 1. Distribution Drift
        distribution_drift = False
        drift_reason = "no_drift"

        if isinstance(reference_data, dict) and isinstance(current_data, dict):
            # Feature-level drift tracking
            self.feature_drifts = {}
            for feature, ref_val in reference_data.items():
                if feature in current_data:
                    cur_val = current_data[feature]
                    _, p_val = calculate_ks_test(ref_val, cur_val)
                    psi = calculate_psi(ref_val, cur_val)

                    is_drifted = (psi > self.psi_threshold) or (p_val < self.ks_p_value_threshold)
                    if is_drifted:
                        self.feature_drifts[feature] = {"psi": psi, "p_value": p_val}
                        distribution_drift = True
                        drift_reason = f"feature_drift:{feature}"
        else:
            # Single-series fallback
            # We need to cast to help mypy understand these are np.ndarray here
            ref_arr = cast(np.ndarray[Any, np.dtype[np.float64]], reference_data)
            cur_arr = cast(np.ndarray[Any, np.dtype[np.float64]], current_data)
            
            _, p_value = calculate_ks_test(ref_arr, cur_arr)
            psi_score = calculate_psi(ref_arr, cur_arr)
            distribution_drift = (psi_score > self.psi_threshold) or (
                p_value < self.ks_p_value_threshold
            )
            if distribution_drift:
                drift_reason = "distribution_drift"

        # 2. Performance Drift
        perf_degraded = False
        if current_perf is not None:
            perf_degraded = self.performance_monitor.detect_drift(current_perf)
            self.performance_monitor.add_metric(current_perf)

        decision = bool(distribution_drift or perf_degraded)
        reason = (
            drift_reason
            if distribution_drift
            else ("performance_degraded" if perf_degraded else "no_drift")
        )

        logger.info(
            "drift_trigger_evaluation",
            decision=decision,
            reason=reason,
            feature_drifts=list(self.feature_drifts.keys()),
            current_perf=current_perf,
        )

        return decision, reason
