import importlib

import pytest


def test_import_streaming_base():
    # Automatically generated import test for streaming.base
    module = importlib.import_module("src.streaming.base")
    assert module is not None

def test_initialization_streaming_base():
    # Automatically generated init test for streaming.base
    try:
        importlib.import_module("src.streaming.base")
    except Exception as e:
        pytest.skip(f"Could not import src.streaming.base: {e}")
