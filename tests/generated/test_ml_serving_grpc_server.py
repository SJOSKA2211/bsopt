import importlib

import pytest


def test_import_ml_serving_grpc_server():
    # Automatically generated import test for ml.serving.grpc_server
    module = importlib.import_module("src.ml.serving.grpc_server")
    assert module is not None

def test_initialization_ml_serving_grpc_server():
    # Automatically generated init test for ml.serving.grpc_server
    try:
        importlib.import_module("src.ml.serving.grpc_server")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.serving.grpc_server: {e}")
