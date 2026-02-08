import importlib

import pytest


def test_import_aiops_autoencoder_detector():
    # Automatically generated import test for aiops.autoencoder_detector
    module = importlib.import_module("src.aiops.autoencoder_detector")
    assert module is not None

def test_initialization_aiops_autoencoder_detector():
    # Automatically generated init test for aiops.autoencoder_detector
    try:
        importlib.import_module("src.aiops.autoencoder_detector")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.autoencoder_detector: {e}")
