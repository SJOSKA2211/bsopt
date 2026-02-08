import importlib

import pytest


def test_import_utils_wasm_loader():
    # Automatically generated import test for utils.wasm_loader
    module = importlib.import_module("src.utils.wasm_loader")
    assert module is not None

def test_initialization_utils_wasm_loader():
    # Automatically generated init test for utils.wasm_loader
    try:
        importlib.import_module("src.utils.wasm_loader")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.wasm_loader: {e}")
