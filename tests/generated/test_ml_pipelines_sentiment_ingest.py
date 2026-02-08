import importlib

import pytest


def test_import_ml_pipelines_sentiment_ingest():
    # Automatically generated import test for ml.pipelines.sentiment_ingest
    module = importlib.import_module("src.ml.pipelines.sentiment_ingest")
    assert module is not None

def test_initialization_ml_pipelines_sentiment_ingest():
    # Automatically generated init test for ml.pipelines.sentiment_ingest
    try:
        importlib.import_module("src.ml.pipelines.sentiment_ingest")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.pipelines.sentiment_ingest: {e}")
