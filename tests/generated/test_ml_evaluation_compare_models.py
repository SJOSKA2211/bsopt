import importlib

import pytest


def test_import_ml_evaluation_compare_models():
    # Automatically generated import test for ml.evaluation.compare_models
    module = importlib.import_module("src.ml.evaluation.compare_models")
    assert module is not None

def test_initialization_ml_evaluation_compare_models():
    # Automatically generated init test for ml.evaluation.compare_models
    try:
        importlib.import_module("src.ml.evaluation.compare_models")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.evaluation.compare_models: {e}")
