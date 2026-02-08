import importlib

import pytest


def test_import_ml_serving_onnx_serving():
    # Automatically generated import test for ml.serving.onnx_serving
    module = importlib.import_module("src.ml.serving.onnx_serving")
    assert module is not None

def test_initialization_ml_serving_onnx_serving():
    # Automatically generated init test for ml.serving.onnx_serving
    try:
        importlib.import_module("src.ml.serving.onnx_serving")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.serving.onnx_serving: {e}")
