import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.streaming.mesh_producer import MarketDataMeshProducer
import os

@pytest.mark.asyncio
async def test_mesh_producer_calls_kafka():
    # Mock SchemaRegistry and Serializer
    with patch("src.streaming.mesh_producer.SchemaRegistryClient") as mock_sr:
        with patch("src.streaming.mesh_producer.AvroSerializer") as mock_avro:
            with patch("src.streaming.mesh_producer.Producer") as mock_prod:
                mock_avro.return_value = lambda data, ctx: b"serialized_data"
                producer = MarketDataMeshProducer(
                    bootstrap_servers="localhost:9092",
                    schema_registry_url="http://localhost:8081"
                )
                
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
                
                assert mock_prod.return_value.produce.called
                args, kwargs = mock_prod.return_value.produce.call_args
                assert kwargs["topic"] == "market-data-mesh"
                assert kwargs["key"] == b"SCOM"
                assert kwargs["value"] == b"serialized_data"

def test_mesh_producer_delivery_callback():
    with patch("src.streaming.mesh_producer.SchemaRegistryClient"):
        with patch("src.streaming.mesh_producer.AvroSerializer"):
            with patch("src.streaming.mesh_producer.Producer"):
                producer = MarketDataMeshProducer()
                # Test success
                msg = MagicMock()
                msg.topic.return_value = "test"
                msg.partition.return_value = 0
                producer._delivery_callback(None, msg)
                
                # Test error
                producer._delivery_callback("Error", msg)

@pytest.mark.asyncio
async def test_mesh_producer_exception_handling():
    with patch("src.streaming.mesh_producer.SchemaRegistryClient"):
        with patch("src.streaming.mesh_producer.AvroSerializer") as mock_avro:
            with patch("src.streaming.mesh_producer.Producer") as mock_prod:
                mock_avro.return_value = lambda data, ctx: b"serialized_data"
                mock_prod.return_value.produce.side_effect = Exception("Kafka Error")
                
                producer = MarketDataMeshProducer()
                # Should not raise exception
                await producer.produce_mesh_data({"symbol": "SCOM"})

def test_mesh_producer_flush():
    with patch("src.streaming.mesh_producer.SchemaRegistryClient"):
        with patch("src.streaming.mesh_producer.AvroSerializer"):
            with patch("src.streaming.mesh_producer.Producer") as mock_prod:
                producer = MarketDataMeshProducer()
                producer.flush()
                assert mock_prod.return_value.flush.called