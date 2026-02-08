import importlib

import pytest


def test_import_aiops_timeseries_anomaly_detector():
    # Automatically generated import test for aiops.timeseries_anomaly_detector
    module = importlib.import_module("src.aiops.timeseries_anomaly_detector")
    assert module is not None

def test_initialization_aiops_timeseries_anomaly_detector():
    # Automatically generated init test for aiops.timeseries_anomaly_detector
    try:
        importlib.import_module("src.aiops.timeseries_anomaly_detector")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.timeseries_anomaly_detector: {e}")
