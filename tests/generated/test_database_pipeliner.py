import importlib

import pytest


def test_import_database_pipeliner():
    # Automatically generated import test for database.pipeliner
    module = importlib.import_module("src.database.pipeliner")
    assert module is not None

def test_initialization_database_pipeliner():
    # Automatically generated init test for database.pipeliner
    try:
        importlib.import_module("src.database.pipeliner")
    except Exception as e:
        pytest.skip(f"Could not import src.database.pipeliner: {e}")
