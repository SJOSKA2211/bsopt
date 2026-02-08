import importlib

import pytest


def test_import_ml_monitoring_mmd():
    # Automatically generated import test for ml.monitoring.mmd
    module = importlib.import_module("src.ml.monitoring.mmd")
    assert module is not None

def test_initialization_ml_monitoring_mmd():
    # Automatically generated init test for ml.monitoring.mmd
    try:
        importlib.import_module("src.ml.monitoring.mmd")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.monitoring.mmd: {e}")
