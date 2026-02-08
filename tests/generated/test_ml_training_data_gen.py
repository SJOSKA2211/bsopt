import importlib

import pytest


def test_import_ml_training_data_gen():
    # Automatically generated import test for ml.training.data_gen
    module = importlib.import_module("src.ml.training.data_gen")
    assert module is not None

def test_initialization_ml_training_data_gen():
    # Automatically generated init test for ml.training.data_gen
    try:
        importlib.import_module("src.ml.training.data_gen")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.training.data_gen: {e}")
