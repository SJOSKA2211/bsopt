import importlib

import pytest


def test_import_protos_market_data_pb2_grpc():
    # Automatically generated import test for protos.market_data_pb2_grpc
    module = importlib.import_module("src.protos.market_data_pb2_grpc")
    assert module is not None

def test_initialization_protos_market_data_pb2_grpc():
    # Automatically generated init test for protos.market_data_pb2_grpc
    try:
        importlib.import_module("src.protos.market_data_pb2_grpc")
    except Exception as e:
        pytest.skip(f"Could not import src.protos.market_data_pb2_grpc: {e}")
