import unittest
import numpy as np
from src.ml.monitoring.mmd import calculate_mmd, MultivariateDriftDetector

class TestMMD(unittest.TestCase):
    def test_calculate_mmd_same_dist(self):
        # Samples from same distribution should have low MMD
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)
        mmd = calculate_mmd(x, y)
        self.assertLess(mmd, 0.5)

    def test_calculate_mmd_diff_dist(self):
        # Samples from different distributions should have higher MMD
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5) + 5.0
        mmd = calculate_mmd(x, y)
        self.assertGreater(mmd, 0.2)

    def test_multivariate_drift_detector(self):
        detector = MultivariateDriftDetector(threshold=0.1)
        baseline = np.random.randn(100, 5)
        current_no_drift = np.random.randn(100, 5)
        current_drift = np.random.randn(100, 5) + 2.0
        
        is_drifted_no, mmd_no = detector.detect_drift(baseline, current_no_drift)
        is_drifted_yes, mmd_yes = detector.detect_drift(baseline, current_drift)
        
        self.assertFalse(is_drifted_no)
        self.assertTrue(is_drifted_yes)
        self.assertGreater(mmd_yes, mmd_no)

if __name__ == '__main__':
    unittest.main()
