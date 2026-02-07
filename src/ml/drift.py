from collections import deque

import numpy as np
import structlog
from scipy.stats import ks_2samp

from src.shared.observability import (
    DATA_DRIFT_SCORE,
    KS_TEST_SCORE,
    PERFORMANCE_DRIFT_ALERT,
)

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
    ):
        self.history = deque(maxlen=window_size)
        self.threshold = threshold
        self.window_size = window_size
        self.higher_is_better = higher_is_better

    def add_metric(self, value: float):
        """Adds a new performance metric to the historical baseline."""
        self.history.append(value)

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
            logger.info(
                "performance_check_passed", baseline=baseline, current=current_value
            )

        return bool(is_drifted)


def calculate_ks_test(
    expected: np.ndarray, actual: np.ndarray | list
) -> tuple[float, float]:
    """
    Calculates the Kolmogorov-Smirnov (KS) test between two distributions.

    Args:
        expected: Reference dataset (e.g., training data).
        actual: Current dataset (e.g., production data).

    Returns:
        Tuple[float, float]: The KS statistic and the p-value.
    """
    logger.info("ks_test_calculation_started")

    expected = np.array(expected)
    actual = np.array(actual)

    statistic, p_value = ks_2samp(expected, actual)

    # Emit Prometheus metric
    KS_TEST_SCORE.set(p_value)

    logger.info("ks_test_calculation_completed", statistic=statistic, p_value=p_value)

    return float(statistic), float(p_value)


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray | list, buckets: int = 10
) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.

    PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))

    Args:
        expected: Reference dataset (e.g., training data).
        actual: Current dataset (e.g., production data).
        buckets: Number of bins for discretization.

    Returns:
        float: The PSI score.
    """
    logger.info("psi_calculation_started", buckets=buckets)

    expected = np.array(expected)
    actual = np.array(actual)

    def scale_range(input_data, min_val, max_val):
        """Discretize the input data into buckets."""
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
        # Handle values exactly at max_val by putting them in the last bucket
        counts, _ = np.histogram(input_data, bins=breakpoints)
        return counts

    # Define range based on the union of both datasets
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    expected_counts = scale_range(expected, min_val, max_val)
    actual_counts = scale_range(actual, min_val, max_val)

    # Convert to percentages
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Handle zero counts by adding a small epsilon to avoid division by zero and log of zero
    epsilon = 1e-6
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    psi_score = np.sum(psi_values)

    # Emit Prometheus metric
    DATA_DRIFT_SCORE.set(psi_score)

    logger.info("psi_calculation_completed", psi_score=psi_score)

    return float(psi_score)


class DriftTrigger:
    """
    Evaluates multiple drift and performance signals to decide if model
    retraining should be triggered.
    """

    def __init__(self, config: dict):
        self.psi_threshold = config.get("psi_threshold", 0.1)
        self.ks_p_value_threshold = config.get("ks_p_value_threshold", 0.05)
        self.perf_threshold = config.get("perf_threshold", 0.05)
        self.force_train = config.get("force_train", False)

        # Use existing PerformanceDriftMonitor
        self.performance_monitor = PerformanceDriftMonitor(
            window_size=config.get("perf_window", 10),
            threshold=self.perf_threshold,
            higher_is_better=config.get("perf_higher_is_better", True),
        )

    def should_retrain(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        current_perf: float | None,
    ) -> tuple[bool, str]:
        """
        Determines if retraining is necessary based on data distribution shift
        and performance degradation.

        Returns:
            Tuple[bool, str]: (Decision, Reason)
        """
        if self.force_train:
            return True, "force_train"

        # 1. Distribution Drift (PSI & KS)
        statistic, p_value = calculate_ks_test(reference_data, current_data)
        psi_score = calculate_psi(reference_data, current_data)

        distribution_drift = (psi_score > self.psi_threshold) or (
            p_value < self.ks_p_value_threshold
        )

        # 2. Performance Drift
        perf_degraded = False
        if current_perf is not None:
            perf_degraded = self.performance_monitor.detect_drift(current_perf)
            self.performance_monitor.add_metric(current_perf)

        decision = bool(distribution_drift or perf_degraded)

        reason = "no_drift"
        if distribution_drift:
            reason = "distribution_drift"
        elif perf_degraded:
            reason = "performance_degraded"

        logger.info(
            "drift_trigger_evaluation",
            decision=decision,
            reason=reason,
            psi_score=psi_score,
            ks_p_value=p_value,
            current_perf=current_perf,
        )

        return decision, reason
