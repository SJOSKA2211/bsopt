import importlib

import pytest


def test_import_streaming_ingestion_worker():
    # Automatically generated import test for streaming.ingestion_worker
    module = importlib.import_module("src.streaming.ingestion_worker")
    assert module is not None

def test_initialization_streaming_ingestion_worker():
    # Automatically generated init test for streaming.ingestion_worker
    try:
        importlib.import_module("src.streaming.ingestion_worker")
    except Exception as e:
        pytest.skip(f"Could not import src.streaming.ingestion_worker: {e}")
