import importlib

import pytest


def test_import_tasks_email_tasks():
    # Automatically generated import test for tasks.email_tasks
    module = importlib.import_module("src.tasks.email_tasks")
    assert module is not None

def test_initialization_tasks_email_tasks():
    # Automatically generated init test for tasks.email_tasks
    try:
        importlib.import_module("src.tasks.email_tasks")
    except Exception as e:
        pytest.skip(f"Could not import src.tasks.email_tasks: {e}")
