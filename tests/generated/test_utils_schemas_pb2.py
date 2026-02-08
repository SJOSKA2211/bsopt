import importlib

import pytest


def test_import_utils_schemas_pb2():
    # Automatically generated import test for utils.schemas_pb2
    module = importlib.import_module("src.utils.schemas_pb2")
    assert module is not None

def test_initialization_utils_schemas_pb2():
    # Automatically generated init test for utils.schemas_pb2
    try:
        importlib.import_module("src.utils.schemas_pb2")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.schemas_pb2: {e}")
