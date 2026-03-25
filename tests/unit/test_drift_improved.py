import pytest

import numpy as np

from src.ml.drift import (
    DriftTrigger,
    PerformanceDriftMonitor,
    calculate_ks_test,
    calculate_psi,
)

class TestDrift:
    def test_performance_drift_monitor(self):
        monitor = PerformanceDriftMonitor(window_size=3, threshold=0.1, higher_is_better=True)
        # Add baseline
        monitor.add_metric(0.9)
        monitor.add_metric(0.92)
        monitor.add_metric(0.91)

        # Test no drift
        self.assertFalse(
            monitor.detect_drift(0.85)
        )  # Baseline avg is 0.91. 0.85 is within 0.1? No, 0.91 - 0.1 = 0.81.

        # Test drift
        assert monitor.detect_drift(0.7)  # 0.7 < 0.81

    def test_calculate_ks_test(self):
        expected = np.random.randn(100)
        actual = np.random.randn(100) + 5.0
        stat, p_val = calculate_ks_test(expected, actual)
        self.assertLess(p_val, 0.05)

    def test_calculate_psi(self):
        expected = np.random.randn(100)
        actual = np.random.randn(100) + 1.0
        psi = calculate_psi(expected, actual)
        self.assertGreater(psi, 0)

    def test_drift_trigger(self):
        config = {"psi_threshold": 0.1, "force_train": False}
        trigger = DriftTrigger(config)
        ref = np.random.randn(100)
        curr = np.random.randn(100)

        decision, reason = trigger.should_retrain(ref, curr, 0.9)
        assert isinstance(decision, bool)
