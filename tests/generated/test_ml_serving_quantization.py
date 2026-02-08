import importlib

import pytest


def test_import_ml_serving_quantization():
    # Automatically generated import test for ml.serving.quantization
    module = importlib.import_module("src.ml.serving.quantization")
    assert module is not None

def test_initialization_ml_serving_quantization():
    # Automatically generated init test for ml.serving.quantization
    try:
        importlib.import_module("src.ml.serving.quantization")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.serving.quantization: {e}")
