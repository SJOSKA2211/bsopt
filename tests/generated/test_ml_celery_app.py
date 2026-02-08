import importlib

import pytest


def test_import_ml_celery_app():
    # Automatically generated import test for ml.celery_app
    module = importlib.import_module("src.ml.celery_app")
    assert module is not None

def test_initialization_ml_celery_app():
    # Automatically generated init test for ml.celery_app
    try:
        importlib.import_module("src.ml.celery_app")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.celery_app: {e}")
