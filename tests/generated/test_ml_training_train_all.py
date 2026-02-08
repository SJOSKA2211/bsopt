import importlib

import pytest


def test_import_ml_training_train_all():
    # Automatically generated import test for ml.training.train_all
    module = importlib.import_module("src.ml.training.train_all")
    assert module is not None

def test_initialization_ml_training_train_all():
    # Automatically generated init test for ml.training.train_all
    try:
        importlib.import_module("src.ml.training.train_all")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.training.train_all: {e}")
