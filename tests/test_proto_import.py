import pytest

def test_generated_protos_import():
    """Test that generated protobuf modules can be imported."""
    try:
        from src.protos import market_data_pb2
        assert market_data_pb2 is not None
    except ImportError:
        pytest.fail("Generated protobuf modules not found. Build system implementation required.")
