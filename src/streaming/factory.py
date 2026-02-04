from enum import Enum
from typing import Optional, Dict, Any
from .base import Producer
from .kafka_producer import MarketDataProducer as KafkaProducer
from .zmq_producer import ZMQMarketDataProducer
from src.config import settings

class StreamingBackend(str, Enum):
    KAFKA = "kafka"
    ZMQ = "zmq"

class StreamingFactory:
    """
    Factory to create streaming producers based on backend type.
    Enforces standardized architecture.
    """
    @staticmethod
    def get_producer(backend: StreamingBackend, **kwargs) -> Producer:
        """
        Get a producer instance for the specified backend.
        
        Args:
            backend: StreamingBackend.KAFKA or StreamingBackend.ZMQ
            **kwargs: Configuration arguments for the producer
        """
        if backend == StreamingBackend.ZMQ:
            endpoint = kwargs.get("endpoint", "tcp://*:5555")
            return ZMQMarketDataProducer(endpoint=endpoint)
        
        elif backend == StreamingBackend.KAFKA:
            bootstrap_servers = kwargs.get("bootstrap_servers", settings.RABBITMQ_URL if "kafka" in settings.RABBITMQ_URL else "localhost:9092") # fallback logic
            # Actually use defaults from kafka_producer if not provided
            return KafkaProducer(
                bootstrap_servers=kwargs.get("bootstrap_servers", "kafka-1:9092,kafka-2:9092,kafka-3:9092"),
                schema_registry_url=kwargs.get("schema_registry_url", "http://schema-registry:8081")
            )
        
        raise ValueError(f"Unsupported streaming backend: {backend}")
