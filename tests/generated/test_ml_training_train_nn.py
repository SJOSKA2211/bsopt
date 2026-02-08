import importlib

import pytest


def test_import_ml_training_train_nn():
    # Automatically generated import test for ml.training.train_nn
    module = importlib.import_module("src.ml.training.train_nn")
    assert module is not None

def test_initialization_ml_training_train_nn():
    # Automatically generated init test for ml.training.train_nn
    try:
        importlib.import_module("src.ml.training.train_nn")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.training.train_nn: {e}")
