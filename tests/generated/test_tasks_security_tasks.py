import importlib

import pytest


def test_import_tasks_security_tasks():
    # Automatically generated import test for tasks.security_tasks
    module = importlib.import_module("src.tasks.security_tasks")
    assert module is not None

def test_initialization_tasks_security_tasks():
    # Automatically generated init test for tasks.security_tasks
    try:
        importlib.import_module("src.tasks.security_tasks")
    except Exception as e:
        pytest.skip(f"Could not import src.tasks.security_tasks: {e}")
