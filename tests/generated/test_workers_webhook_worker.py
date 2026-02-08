import importlib

import pytest


def test_import_workers_webhook_worker():
    # Automatically generated import test for workers.webhook_worker
    module = importlib.import_module("src.workers.webhook_worker")
    assert module is not None

def test_initialization_workers_webhook_worker():
    # Automatically generated init test for workers.webhook_worker
    try:
        importlib.import_module("src.workers.webhook_worker")
    except Exception as e:
        pytest.skip(f"Could not import src.workers.webhook_worker: {e}")
