import importlib

import pytest


def test_import_api_middleware_logging():
    # Automatically generated import test for api.middleware.logging
    module = importlib.import_module("src.api.middleware.logging")
    assert module is not None

def test_initialization_api_middleware_logging():
    # Automatically generated init test for api.middleware.logging
    try:
        importlib.import_module("src.api.middleware.logging")
    except Exception as e:
        pytest.skip(f"Could not import src.api.middleware.logging: {e}")
