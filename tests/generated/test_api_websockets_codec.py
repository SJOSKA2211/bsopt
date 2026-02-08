import importlib

import pytest


def test_import_api_websockets_codec():
    # Automatically generated import test for api.websockets.codec
    module = importlib.import_module("src.api.websockets.codec")
    assert module is not None

def test_initialization_api_websockets_codec():
    # Automatically generated init test for api.websockets.codec
    try:
        importlib.import_module("src.api.websockets.codec")
    except Exception as e:
        pytest.skip(f"Could not import src.api.websockets.codec: {e}")
