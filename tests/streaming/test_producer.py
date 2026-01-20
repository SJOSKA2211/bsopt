import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.streaming.producer import MarketDataProducer
import os

@pytest.fixture
def mock_producer_deps():
    with patch("src.streaming.producer.Producer") as mock_p, \
         patch("src.streaming.producer.SchemaRegistryClient") as mock_sr, \
         patch("src.streaming.producer.AvroSerializer") as mock_ser, \
         patch("builtins.open", mock_open(read_data='{"type": "record", "name": "MarketData", "fields": []}')):
        yield mock_p, mock_sr, mock_ser

def test_producer_init(mock_producer_deps):
    mock_p, mock_sr, mock_ser = mock_producer_deps
    producer = MarketDataProducer()
    assert producer.config['client.id'] == 'market-data-producer'
    mock_p.assert_called_once()

@pytest.mark.asyncio
async def test_produce_market_data(mock_producer_deps):
    mock_p, mock_sr, mock_ser = mock_producer_deps
    producer = MarketDataProducer()
    mock_ser.return_value.return_value = b"serialized"
    
    await producer.produce_market_data("topic", {"data": 1}, key="key")
    producer.producer.produce.assert_called_once()

@pytest.mark.asyncio
async def test_produce_error(mock_producer_deps):
    mock_p, mock_sr, mock_ser = mock_producer_deps
    producer = MarketDataProducer()
    producer.producer.produce.side_effect = Exception("error")
    
    with pytest.raises(Exception):
        await producer.produce_market_data("topic", {})

def test_delivery_callback(mock_producer_deps):
    producer = MarketDataProducer()
    msg = MagicMock()
    producer._delivery_callback(None, msg)
    producer._delivery_callback("error", msg)

def test_flush(mock_producer_deps):
    producer = MarketDataProducer()
    producer.flush()
    producer.producer.flush.assert_called_once()
