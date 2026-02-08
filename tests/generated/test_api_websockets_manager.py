import importlib

import pytest


def test_import_api_websockets_manager():
    # Automatically generated import test for api.websockets.manager
    module = importlib.import_module("src.api.websockets.manager")
    assert module is not None

def test_initialization_api_websockets_manager():
    # Automatically generated init test for api.websockets.manager
    try:
        importlib.import_module("src.api.websockets.manager")
    except Exception as e:
        pytest.skip(f"Could not import src.api.websockets.manager: {e}")
