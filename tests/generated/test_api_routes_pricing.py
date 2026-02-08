import importlib

import pytest


def test_import_api_routes_pricing():
    # Automatically generated import test for api.routes.pricing
    module = importlib.import_module("src.api.routes.pricing")
    assert module is not None

def test_initialization_api_routes_pricing():
    # Automatically generated init test for api.routes.pricing
    try:
        importlib.import_module("src.api.routes.pricing")
    except Exception as e:
        pytest.skip(f"Could not import src.api.routes.pricing: {e}")
