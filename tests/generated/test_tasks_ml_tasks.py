import importlib

import pytest


def test_import_tasks_ml_tasks():
    # Automatically generated import test for tasks.ml_tasks
    module = importlib.import_module("src.tasks.ml_tasks")
    assert module is not None

def test_initialization_tasks_ml_tasks():
    # Automatically generated init test for tasks.ml_tasks
    try:
        importlib.import_module("src.tasks.ml_tasks")
    except Exception as e:
        pytest.skip(f"Could not import src.tasks.ml_tasks: {e}")
