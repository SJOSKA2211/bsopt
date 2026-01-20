import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.streaming.mesh_producer import MarketDataMeshProducer
import os

@pytest.mark.asyncio
async def test_mesh_producer_calls_kafka(mocker):
    # Mock dependencies inside the module
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"type": "record", "name": "MarketDataMesh"}'))
    
    mock_sr_cls = mocker.patch("src.streaming.mesh_producer.SchemaRegistryClient")
    mock_avro_cls = mocker.patch("src.streaming.mesh_producer.AvroSerializer")
    mock_prod_cls = mocker.patch("src.streaming.mesh_producer.Producer")
    
    # Setup instances
    mock_prod_instance = MagicMock()
    mock_prod_cls.return_value = mock_prod_instance
    
    mock_serializer_instance = MagicMock()
    mock_serializer_instance.return_value = b"serialized_data"
    mock_avro_cls.return_value = mock_serializer_instance
    
    producer = MarketDataMeshProducer(
        bootstrap_servers="localhost:9092",
        schema_registry_url="http://localhost:8081"
    )
    
    print(f"DEBUG: producer.producer type: {type(producer.producer)}")
    assert producer.producer is mock_prod_instance

    data = {
        "symbol": "SCOM",
        "timestamp": "2026-01-14T10:00:00",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
        "market": "NSE",
        "source_type": "scraper"
    }

    await producer.produce_mesh_data(data)

    assert mock_prod_instance.produce.called
    args, kwargs = mock_prod_instance.produce.call_args
    assert kwargs["topic"] == "market-data-mesh"
    assert kwargs["key"] == b"SCOM"
    assert kwargs["value"] == b"serialized_data"

def test_mesh_producer_delivery_callback(mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("src.streaming.mesh_producer.SchemaRegistryClient")
    mocker.patch("src.streaming.mesh_producer.AvroSerializer")
    mocker.patch("src.streaming.mesh_producer.Producer")
    
    producer = MarketDataMeshProducer()
    msg = MagicMock()
    msg.topic.return_value = "test"
    msg.partition.return_value = 0
    producer._delivery_callback(None, msg)
    producer._delivery_callback("Error", msg)

@pytest.mark.asyncio
async def test_mesh_producer_exception_handling(mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("src.streaming.mesh_producer.SchemaRegistryClient")
    mock_avro_cls = mocker.patch("src.streaming.mesh_producer.AvroSerializer")
    mock_prod_cls = mocker.patch("src.streaming.mesh_producer.Producer")
    
    mock_prod_instance = MagicMock()
    mock_prod_cls.return_value = mock_prod_instance
    mock_prod_instance.produce.side_effect = Exception("Kafka Error")
    
    mock_serializer_instance = MagicMock()
    mock_serializer_instance.return_value = b"data"
    mock_avro_cls.return_value = mock_serializer_instance
    
    producer = MarketDataMeshProducer()
    await producer.produce_mesh_data({"symbol": "SCOM"})

def test_mesh_producer_flush(mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch("src.streaming.mesh_producer.SchemaRegistryClient")
    mocker.patch("src.streaming.mesh_producer.AvroSerializer")
    mock_prod_cls = mocker.patch("src.streaming.mesh_producer.Producer")
    
    mock_prod_instance = MagicMock()
    mock_prod_cls.return_value = mock_prod_instance
    
    producer = MarketDataMeshProducer()
    producer.flush()
    assert mock_prod_instance.flush.called