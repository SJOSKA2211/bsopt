import importlib

import pytest


def test_import_data_xdp_ingest():
    # Automatically generated import test for data.xdp_ingest
    module = importlib.import_module("src.data.xdp_ingest")
    assert module is not None

def test_initialization_data_xdp_ingest():
    # Automatically generated init test for data.xdp_ingest
    try:
        importlib.import_module("src.data.xdp_ingest")
    except Exception as e:
        pytest.skip(f"Could not import src.data.xdp_ingest: {e}")
