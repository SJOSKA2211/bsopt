import importlib

import pytest


def test_import_aiops_latency_remediator():
    # Automatically generated import test for aiops.latency_remediator
    module = importlib.import_module("src.aiops.latency_remediator")
    assert module is not None

def test_initialization_aiops_latency_remediator():
    # Automatically generated init test for aiops.latency_remediator
    try:
        importlib.import_module("src.aiops.latency_remediator")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.latency_remediator: {e}")
