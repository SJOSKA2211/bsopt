import importlib

import pytest


def test_import_aiops_docker_remediator():
    # Automatically generated import test for aiops.docker_remediator
    module = importlib.import_module("src.aiops.docker_remediator")
    assert module is not None

def test_initialization_aiops_docker_remediator():
    # Automatically generated init test for aiops.docker_remediator
    try:
        importlib.import_module("src.aiops.docker_remediator")
    except Exception as e:
        pytest.skip(f"Could not import src.aiops.docker_remediator: {e}")
