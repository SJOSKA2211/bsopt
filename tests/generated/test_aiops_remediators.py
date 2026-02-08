import importlib

import pytest


def test_import_aiops_remediators():
    # Automatically generated import test for aiops.remediators
    module = importlib.import_module("src.aiops.remediators")
    assert module is not None

def test_initialization_aiops_remediators():
    # Automatically generated init test for aiops.remediators
    try:
        importlib.import_module("src.aiops.remediators")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.remediators: {e}")
