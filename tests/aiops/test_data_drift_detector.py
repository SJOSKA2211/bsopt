import unittest
from unittest.mock import call, patch

import numpy as np

from src.aiops.data_drift_detector import DataDriftDetector


class TestDataDriftDetector(unittest.TestCase):
    def setUp(self):
        self.patcher_psi = patch("src.aiops.data_drift_detector.calculate_psi")
        self.patcher_ks = patch("src.aiops.data_drift_detector.calculate_ks_test")
        self.patcher_psi_gauge = patch("src.aiops.data_drift_detector.PSI_DRIFT_STATUS")
        self.patcher_ks_gauge = patch("src.aiops.data_drift_detector.KS_DRIFT_STATUS")
        self.patcher_overall_gauge = patch(
            "src.aiops.data_drift_detector.OVERALL_DRIFT_STATUS"
        )
        self.patcher_logger = patch("src.aiops.data_drift_detector.logger")

        self.mock_psi = self.patcher_psi.start()
        self.mock_ks = self.patcher_ks.start()
        self.mock_psi_gauge = self.patcher_psi_gauge.start()
        self.mock_ks_gauge = self.patcher_ks_gauge.start()
        self.mock_overall_gauge = self.patcher_overall_gauge.start()
        self.mock_logger = self.patcher_logger.start()

        self.detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        self.assertEqual(self.detector.psi_threshold, 0.1)
        self.assertEqual(self.detector.ks_threshold, 0.05)

    def test_detect_drift_empty_data(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.detector.detect_drift(np.array([]), np.array([1, 2]))
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.detector.detect_drift(np.array([1, 2]), np.array([]))
        self.mock_logger.info.assert_not_called()  # No logger calls on error

    def test_detect_drift_mismatched_dimensions(self):
        ref = np.random.rand(10, 2)
        curr = np.random.rand(10, 3)
        with self.assertRaisesRegex(
            ValueError, "must have the same number of features"
        ):
            self.detector.detect_drift(ref, curr)
        self.mock_logger.info.assert_not_called()  # No logger calls on error

    def test_detect_drift_univariate_no_drift(self):
        ref = np.random.rand(10, 1)
        curr = np.random.rand(10, 1)

        # Mock no drift
        self.mock_psi.return_value = 0.05  # < 0.1
        self.mock_ks.return_value = (0.0, 0.1)  # p-value > 0.05

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertFalse(drift)
        self.assertFalse(info["overall_drift_detected"])
        self.mock_psi.assert_called_once()
        self.mock_ks.assert_called_once()
        self.mock_psi_gauge.labels.assert_called_once_with(feature="overall")
        self.mock_psi_gauge.labels.return_value.set.assert_called_once_with(0)
        self.mock_ks_gauge.labels.assert_called_once_with(feature="overall")
        self.mock_ks_gauge.labels.return_value.set.assert_called_once_with(0)
        self.mock_overall_gauge.set.assert_called_once_with(0)
        self.mock_logger.info.assert_called_once_with(
            "data_drift_univariate_check", psi=0.05, ks_p=0.1, drift_detected=False
        )

    def test_detect_drift_univariate_psi_drift(self):
        ref = np.random.rand(10, 1)
        curr = np.random.rand(10, 1)

        # Mock PSI drift
        self.mock_psi.return_value = 0.2  # > 0.1
        self.mock_ks.return_value = (0.0, 0.1)  # No KS drift

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertTrue(drift)
        self.assertTrue(info["overall_drift_detected"])
        self.assertIn("PSI_Drift", info["drift_types"])
        self.mock_psi_gauge.labels.return_value.set.assert_called_once_with(1)
        self.mock_ks_gauge.labels.return_value.set.assert_called_once_with(0)
        self.mock_overall_gauge.set.assert_called_once_with(1)
        self.mock_logger.info.assert_called_once_with(
            "data_drift_univariate_check", psi=0.2, ks_p=0.1, drift_detected=True
        )

    def test_detect_drift_univariate_ks_drift(self):
        ref = np.random.rand(10, 1)
        curr = np.random.rand(10, 1)

        # Mock KS drift
        self.mock_psi.return_value = 0.05
        self.mock_ks.return_value = (0.0, 0.01)  # p-value < 0.05

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertTrue(drift)
        self.assertTrue(info["overall_drift_detected"])
        self.assertIn("KS_Drift", info["drift_types"])
        self.mock_psi_gauge.labels.return_value.set.assert_called_once_with(0)
        self.mock_ks_gauge.labels.return_value.set.assert_called_once_with(1)
        self.mock_overall_gauge.set.assert_called_once_with(1)
        self.mock_logger.info.assert_called_once_with(
            "data_drift_univariate_check", psi=0.05, ks_p=0.01, drift_detected=True
        )

    def test_detect_drift_multivariate_no_drift(self):
        ref = np.random.rand(10, 2)
        curr = np.random.rand(10, 2)

        self.mock_psi.side_effect = [0.05, 0.05]  # Both features no PSI drift
        self.mock_ks.side_effect = [(0.0, 0.1), (0.0, 0.1)]  # Both features no KS drift

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertFalse(drift)
        self.assertFalse(info["overall_drift_detected"])
        self.assertEqual(len(info["feature_drifts"]), 0)

        # Verify gauge calls for both features setting to 0
        self.mock_psi_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_psi_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)

        self.mock_ks_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_ks_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)

        # Two calls for logger for each feature check
        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.mock_logger.info.assert_has_calls(
            [
                call(
                    "data_drift_feature_check",
                    feature_index=0,
                    psi=0.05,
                    ks_p=0.1,
                    drift_detected=False,
                ),
                call(
                    "data_drift_feature_check",
                    feature_index=1,
                    psi=0.05,
                    ks_p=0.1,
                    drift_detected=False,
                ),
            ],
            any_order=True,
        )
        self.mock_overall_gauge.set.assert_called_once_with(0)

    def test_detect_drift_multivariate_psi_drift_single_feature(self):
        ref = np.random.rand(10, 2)
        curr = np.random.rand(10, 2)

        # Mock side effects for 2 features
        # Feature 0: No Drift
        # Feature 1: PSI Drift
        self.mock_psi.side_effect = [0.05, 0.2]  # 0: no drift, 1: PSI drift
        self.mock_ks.side_effect = [(0.0, 0.1), (0.0, 0.1)]  # Both no KS drift

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertTrue(drift)
        self.assertTrue(info["overall_drift_detected"])
        self.assertEqual(len(info["feature_drifts"]), 1)
        self.assertEqual(info["feature_drifts"][0]["feature_index"], 1)
        self.assertTrue(info["feature_drifts"][0]["drift_detected"])
        self.assertIn("PSI_Drift", info["feature_drifts"][0]["drift_types"])

        self.mock_psi_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_psi_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(1)

        self.mock_ks_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_ks_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)

        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.mock_logger.info.assert_has_calls(
            [
                call(
                    "data_drift_feature_check",
                    feature_index=0,
                    psi=0.05,
                    ks_p=0.1,
                    drift_detected=False,
                ),
                call(
                    "data_drift_feature_check",
                    feature_index=1,
                    psi=0.2,
                    ks_p=0.1,
                    drift_detected=True,
                ),
            ],
            any_order=True,
        )
        self.mock_overall_gauge.set.assert_called_once_with(1)

    def test_detect_drift_multivariate_ks_drift_single_feature(self):
        ref = np.random.rand(10, 2)
        curr = np.random.rand(10, 2)

        # Feature 0: No Drift
        # Feature 1: KS Drift
        self.mock_psi.side_effect = [0.05, 0.05]  # Both features no PSI drift
        self.mock_ks.side_effect = [
            (0.0, 0.1),
            (0.0, 0.01),
        ]  # 0: no KS drift, 1: KS drift

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertTrue(drift)
        self.assertTrue(info["overall_drift_detected"])
        self.assertIn("KS_Drift", info["feature_drifts"][0]["drift_types"])

        self.mock_psi_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_psi_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)

        self.mock_ks_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_ks_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(1)

        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.mock_logger.info.assert_has_calls(
            [
                call(
                    "data_drift_feature_check",
                    feature_index=0,
                    psi=0.05,
                    ks_p=0.1,
                    drift_detected=False,
                ),
                call(
                    "data_drift_feature_check",
                    feature_index=1,
                    psi=0.05,
                    ks_p=0.01,
                    drift_detected=True,
                ),
            ],
            any_order=True,
        )
        self.mock_overall_gauge.set.assert_called_once_with(1)

    def test_detect_drift_multivariate_both_drifts_single_feature(self):
        ref = np.random.rand(10, 2)
        curr = np.random.rand(10, 2)

        # Feature 0: No Drift
        # Feature 1: Both PSI and KS Drift
        self.mock_psi.side_effect = [0.05, 0.2]  # 0: no drift, 1: PSI drift
        self.mock_ks.side_effect = [
            (0.0, 0.1),
            (0.0, 0.01),
        ]  # 0: no KS drift, 1: KS drift

        drift, info = self.detector.detect_drift(ref, curr)

        self.assertTrue(drift)
        self.assertTrue(info["overall_drift_detected"])
        self.assertEqual(len(info["feature_drifts"]), 1)
        self.assertEqual(info["feature_drifts"][0]["feature_index"], 1)
        self.assertTrue(info["feature_drifts"][0]["drift_detected"])
        self.assertIn("PSI_Drift", info["feature_drifts"][0]["drift_types"])
        self.assertIn("KS_Drift", info["feature_drifts"][0]["drift_types"])

        self.mock_psi_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_psi_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_psi_gauge.labels.return_value.set.assert_any_call(1)

        self.mock_ks_gauge.labels.assert_any_call(feature="feature_0")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(0)
        self.mock_ks_gauge.labels.assert_any_call(feature="feature_1")
        self.mock_ks_gauge.labels.return_value.set.assert_any_call(1)

        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.mock_logger.info.assert_has_calls(
            [
                call(
                    "data_drift_feature_check",
                    feature_index=0,
                    psi=0.05,
                    ks_p=0.1,
                    drift_detected=False,
                ),
                call(
                    "data_drift_feature_check",
                    feature_index=1,
                    psi=0.2,
                    ks_p=0.01,
                    drift_detected=True,
                ),
            ],
            any_order=True,
        )
        self.mock_overall_gauge.set.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
