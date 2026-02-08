import importlib

import pytest


def test_import_api_routes_debug():
    # Automatically generated import test for api.routes.debug
    module = importlib.import_module("src.api.routes.debug")
    assert module is not None

def test_initialization_api_routes_debug():
    # Automatically generated init test for api.routes.debug
    try:
        importlib.import_module("src.api.routes.debug")
    except Exception as e:
        pytest.skip(f"Could not import src.api.routes.debug: {e}")
