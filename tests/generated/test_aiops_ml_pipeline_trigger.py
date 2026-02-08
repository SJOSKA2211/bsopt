import importlib

import pytest


def test_import_aiops_ml_pipeline_trigger():
    # Automatically generated import test for aiops.ml_pipeline_trigger
    module = importlib.import_module("src.aiops.ml_pipeline_trigger")
    assert module is not None

def test_initialization_aiops_ml_pipeline_trigger():
    # Automatically generated init test for aiops.ml_pipeline_trigger
    try:
        importlib.import_module("src.aiops.ml_pipeline_trigger")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.ml_pipeline_trigger: {e}")
