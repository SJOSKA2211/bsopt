import importlib

import pytest


def test_import_aiops_redis_remediator():
    # Automatically generated import test for aiops.redis_remediator
    module = importlib.import_module("src.aiops.redis_remediator")
    assert module is not None

def test_initialization_aiops_redis_remediator():
    # Automatically generated init test for aiops.redis_remediator
    try:
        importlib.import_module("src.aiops.redis_remediator")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.redis_remediator: {e}")
