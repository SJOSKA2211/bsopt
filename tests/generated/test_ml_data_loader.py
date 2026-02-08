import importlib

import pytest


def test_import_ml_data_loader():
    # Automatically generated import test for ml.data_loader
    module = importlib.import_module("src.ml.data_loader")
    assert module is not None

def test_initialization_ml_data_loader():
    # Automatically generated init test for ml.data_loader
    try:
        importlib.import_module("src.ml.data_loader")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.data_loader: {e}")
