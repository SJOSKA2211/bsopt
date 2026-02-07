import numpy as np
import pytest

from src.aiops.data_drift_detector import DataDriftDetector  # Assuming this path

# Mock for simplicity if DataDriftDetector doesn't directly use these or needs a wrapper
# For this Red Phase, we just need to test the detector's logic around these scores.


def test_data_drift_detector_init():
    """Test initialization of DataDriftDetector."""
    psi_threshold = 0.1
    ks_threshold = 0.05
    detector = DataDriftDetector(psi_threshold=psi_threshold, ks_threshold=ks_threshold)
    assert detector.psi_threshold == psi_threshold
    assert detector.ks_threshold == ks_threshold


def test_data_drift_detector_no_drift():
    """Test with data showing no significant drift."""
    np.random.seed(42)
    reference_data = np.random.normal(loc=0, scale=1, size=(100, 2))
    current_data = np.random.normal(loc=0.01, scale=1.01, size=(100, 2))  # Slight shift

    detector = DataDriftDetector(psi_threshold=0.5, ks_threshold=0.05)
    drift_detected, drift_info = detector.detect_drift(reference_data, current_data)

    assert not drift_detected
    assert (
        len(drift_info["feature_drifts"]) == 0
    )  # No feature drifts should be detected
    # We can't directly check 'PSI' in drift_info for multivariate, but can check overall status
    assert not drift_info["overall_drift_detected"]


def test_data_drift_detector_psi_drift_detected():
    """Test with data showing significant PSI drift."""
    np.random.seed(42)
    reference_data = np.random.normal(loc=0, scale=1, size=(100, 1))
    current_data = np.random.normal(
        loc=2, scale=1, size=(100, 1)
    )  # Clear shift for PSI

    detector = DataDriftDetector(
        psi_threshold=0.1, ks_threshold=0.001
    )  # Lower KS threshold to focus on PSI
    drift_detected, drift_info = detector.detect_drift(reference_data, current_data)

    assert drift_detected
    assert drift_info["PSI"] >= detector.psi_threshold
    assert "PSI_Drift" in drift_info["drift_types"]


def test_data_drift_detector_ks_drift_detected():
    """Test with data showing significant KS drift."""
    np.random.seed(42)
    reference_data = np.random.normal(loc=0, scale=1, size=(100, 1))
    current_data = np.random.normal(loc=0.5, scale=1, size=(100, 1))  # Shift for KS

    detector = DataDriftDetector(
        psi_threshold=1.0, ks_threshold=0.05
    )  # Higher PSI threshold to focus on KS
    drift_detected, drift_info = detector.detect_drift(reference_data, current_data)

    assert drift_detected
    assert drift_info["KS_P_VALUE"] <= detector.ks_threshold
    assert "KS_Drift" in drift_info["drift_types"]


def test_data_drift_detector_empty_data_raises_error():
    """Test with empty input data."""
    reference_data = np.array([]).reshape(0, 1)
    current_data = np.array([]).reshape(0, 1)
    detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)
    with pytest.raises(ValueError, match="Reference or current data cannot be empty."):
        detector.detect_drift(reference_data, current_data)


def test_data_drift_detector_unequal_dimensions_raises_error():
    """Test with unequal feature dimensions."""
    reference_data = np.random.rand(10, 2)
    current_data = np.random.rand(10, 3)
    detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)
    with pytest.raises(
        ValueError,
        match="Reference and current data must have the same number of features.",
    ):
        detector.detect_drift(reference_data, current_data)


def test_data_drift_detector_multivariate_detection():
    """Test detection on multivariate data with drift in one feature."""
    np.random.seed(42)
    reference_data = np.random.normal(loc=0, scale=1, size=(100, 3))

    # Drift only in the second feature
    current_data = reference_data.copy()
    current_data[:, 1] = np.random.normal(loc=3, scale=1, size=100)

    detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)
    drift_detected, drift_info = detector.detect_drift(reference_data, current_data)

    assert drift_detected
    assert "feature_drifts" in drift_info
    assert len(drift_info["feature_drifts"]) == 1
    assert drift_info["feature_drifts"][0]["feature_index"] == 1
    assert (
        "PSI_Drift" in drift_info["feature_drifts"][0]["drift_types"]
        or "KS_Drift" in drift_info["feature_drifts"][0]["drift_types"]
    )
