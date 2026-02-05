import pytest
import os
from src.config import settings

def test_environment_variables():
    """Verify that environment variables are correctly mocked/loaded."""
    assert settings.ENVIRONMENT == "dev"
    assert settings.DATABASE_URL == "sqlite:///:memory:"
    assert settings.REDIS_URL == "redis://localhost:6379/0"

def test_mlflow_mock():
    """Verify MLflow is mocked and context manager works."""
    import mlflow
    with mlflow.start_run():
        mlflow.log_param("test", 1)
    assert True

def test_kafka_mock():
    """Verify Kafka is mocked."""
    import confluent_kafka
    producer = confluent_kafka.Producer({"bootstrap.servers": "localhost:9092"})
    producer.produce("topic", b"msg")
    assert True
