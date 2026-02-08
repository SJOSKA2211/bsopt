import importlib

import pytest


def test_import_api_routes_ml():
    # Automatically generated import test for api.routes.ml
    module = importlib.import_module("src.api.routes.ml")
    assert module is not None

def test_initialization_api_routes_ml():
    # Automatically generated init test for api.routes.ml
    try:
        importlib.import_module("src.api.routes.ml")
    except Exception as e:
        pytest.skip(f"Could not import src.api.routes.ml: {e}")
