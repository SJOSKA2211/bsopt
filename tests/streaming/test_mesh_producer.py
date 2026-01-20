import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.streaming.mesh_producer import MarketDataMeshProducer
import os

@pytest.fixture
def mock_mesh_producer_deps():
    with patch("src.streaming.mesh_producer.Producer") as mock_p, \
         patch("src.streaming.mesh_producer.SchemaRegistryClient") as mock_sr, \
         patch("src.streaming.mesh_producer.AvroSerializer") as mock_ser, \
         patch("builtins.open", mock_open(read_data='{"type": "record", "name": "Mesh", "fields": []}')):
        yield mock_p, mock_sr, mock_ser

def test_mesh_producer_init(mock_mesh_producer_deps):
    producer = MarketDataMeshProducer()
    assert producer.config['client.id'] == 'market-data-mesh-producer'

@pytest.mark.asyncio
async def test_produce_mesh_data(mock_mesh_producer_deps):
    mock_p, mock_sr, mock_ser = mock_mesh_producer_deps
    producer = MarketDataMeshProducer()
    mock_ser.return_value.return_value = b"mesh_serialized"
    
    await producer.produce_mesh_data({"symbol": "AAPL", "price": 150})
    producer.producer.produce.assert_called_once()

@pytest.mark.asyncio
async def test_produce_mesh_error(mock_mesh_producer_deps):
    producer = MarketDataMeshProducer()
    producer.producer.produce.side_effect = Exception("Mesh error")
    
    with patch("src.streaming.mesh_producer.logger") as mock_logger:
        await producer.produce_mesh_data({"symbol": "AAPL"})
        mock_logger.error.assert_called_with("mesh_kafka_produce_error", error="Mesh error", symbol="AAPL")

def test_delivery_callback(mock_mesh_producer_deps):
    producer = MarketDataMeshProducer()
    msg = MagicMock()
    producer._delivery_callback(None, msg)
    producer._delivery_callback("error", msg)

def test_flush(mock_mesh_producer_deps):
    producer = MarketDataMeshProducer()
    producer.flush()
    producer.producer.flush.assert_called_once()
