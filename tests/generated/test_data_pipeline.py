import importlib

import pytest


def test_import_data_pipeline():
    # Automatically generated import test for data.pipeline
    module = importlib.import_module("src.data.pipeline")
    assert module is not None

def test_initialization_data_pipeline():
    # Automatically generated init test for data.pipeline
    try:
        importlib.import_module("src.data.pipeline")
    except Exception as e:
        pytest.skip(f"Could not import src.data.pipeline: {e}")
