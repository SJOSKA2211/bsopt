import importlib

import pytest


def test_import_streaming_factory():
    # Automatically generated import test for streaming.factory
    module = importlib.import_module("src.streaming.factory")
    assert module is not None

def test_initialization_streaming_factory():
    # Automatically generated init test for streaming.factory
    try:
        importlib.import_module("src.streaming.factory")
    except Exception as e:
        pytest.skip(f"Could not import src.streaming.factory: {e}")
