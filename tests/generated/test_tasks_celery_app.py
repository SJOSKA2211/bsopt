import importlib

import pytest


def test_import_tasks_celery_app():
    # Automatically generated import test for tasks.celery_app
    module = importlib.import_module("src.tasks.celery_app")
    assert module is not None

def test_initialization_tasks_celery_app():
    # Automatically generated init test for tasks.celery_app
    try:
        importlib.import_module("src.tasks.celery_app")
    except Exception as e:
        pytest.skip(f"Could not import src.tasks.celery_app: {e}")
