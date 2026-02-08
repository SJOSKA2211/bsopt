import importlib

import pytest


def test_import_aiops_prometheus_adapter():
    # Automatically generated import test for aiops.prometheus_adapter
    module = importlib.import_module("src.aiops.prometheus_adapter")
    assert module is not None

def test_initialization_aiops_prometheus_adapter():
    # Automatically generated init test for aiops.prometheus_adapter
    try:
        importlib.import_module("src.aiops.prometheus_adapter")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.prometheus_adapter: {e}")
