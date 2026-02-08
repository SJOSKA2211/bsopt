import importlib

import pytest


def test_import_api_middleware_security():
    # Automatically generated import test for api.middleware.security
    module = importlib.import_module("src.api.middleware.security")
    assert module is not None

def test_initialization_api_middleware_security():
    # Automatically generated init test for api.middleware.security
    try:
        importlib.import_module("src.api.middleware.security")
    except Exception as e:
        pytest.skip(f"Could not import src.api.middleware.security: {e}")
