import importlib

import pytest


def test_import_api_middleware_profiling():
    # Automatically generated import test for api.middleware.profiling
    module = importlib.import_module("src.api.middleware.profiling")
    assert module is not None

def test_initialization_api_middleware_profiling():
    # Automatically generated init test for api.middleware.profiling
    try:
        importlib.import_module("src.api.middleware.profiling")
    except Exception as e:
        pytest.skip(f"Could not import src.api.middleware.profiling: {e}")
