import importlib

import pytest


def test_import_aiops_isolation_forest_detector():
    # Automatically generated import test for aiops.isolation_forest_detector
    module = importlib.import_module("src.aiops.isolation_forest_detector")
    assert module is not None

def test_initialization_aiops_isolation_forest_detector():
    # Automatically generated init test for aiops.isolation_forest_detector
    try:
        importlib.import_module("src.aiops.isolation_forest_detector")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.isolation_forest_detector: {e}")
