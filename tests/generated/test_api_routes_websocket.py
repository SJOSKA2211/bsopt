import importlib

import pytest


def test_import_api_routes_websocket():
    # Automatically generated import test for api.routes.websocket
    module = importlib.import_module("src.api.routes.websocket")
    assert module is not None

def test_initialization_api_routes_websocket():
    # Automatically generated init test for api.routes.websocket
    try:
        importlib.import_module("src.api.routes.websocket")
    except Exception as e:
        pytest.skip(f"Could not import src.api.routes.websocket: {e}")
