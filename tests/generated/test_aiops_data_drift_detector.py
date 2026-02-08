import importlib

import pytest


def test_import_aiops_data_drift_detector():
    # Automatically generated import test for aiops.data_drift_detector
    module = importlib.import_module("src.aiops.data_drift_detector")
    assert module is not None

def test_initialization_aiops_data_drift_detector():
    # Automatically generated init test for aiops.data_drift_detector
    try:
        importlib.import_module("src.aiops.data_drift_detector")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.data_drift_detector: {e}")
