import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming MarketDataConsumer will be importable from src.streaming.kafka_consumer
try:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
    from confluent_kafka import KafkaError  # Needed for mocking KafkaError

    from streaming.kafka_consumer import MarketDataConsumer
except ImportError:
    MarketDataConsumer = None
    KafkaError = MagicMock()


CONSUMER_PATH = "src/streaming/kafka_consumer.py"
TEST_BOOTSTRAP_SERVERS = "localhost:9092"
TEST_GROUP_ID = "test-group"
TEST_TOPICS = ["test-topic"]
TEST_MESSAGE_VALUE = '{"symbol": "AAPL", "price": 150.0}'


def test_kafka_consumer_file_exists():
    """
    Test that the kafka_consumer.py file exists.
    """
    assert os.path.exists(
        CONSUMER_PATH
    ), f"MarketDataConsumer file not found at {CONSUMER_PATH}"


def test_market_data_consumer_class_exists():
    """
    Test that the MarketDataConsumer class can be imported.
    This test will fail if the class is not yet defined or importable.
    """
    assert (
        MarketDataConsumer is not None
    ), "MarketDataConsumer class is not defined or importable."


@patch("streaming.kafka_consumer.Consumer")
def test_market_data_consumer_init(mock_consumer):
    """
    Test that MarketDataConsumer constructor initializes Consumer and subscribes to topics.
    """
    mock_consumer_instance = MagicMock()
    mock_consumer.return_value = mock_consumer_instance

    consumer = MarketDataConsumer(TEST_BOOTSTRAP_SERVERS, TEST_GROUP_ID, TEST_TOPICS)

    # Assertions for constructor calls
    mock_consumer.assert_called_once_with(
        {
            "bootstrap.servers": TEST_BOOTSTRAP_SERVERS,
            "group.id": TEST_GROUP_ID,
            "auto.offset.reset": "latest",
            "fetch.min.bytes": 1024,
            "fetch.wait.max.ms": 100,
            "max.partition.fetch.bytes": 1048576,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 10000,
            "heartbeat.interval.ms": 3000,
        }
    )
    mock_consumer_instance.subscribe.assert_called_once_with(TEST_TOPICS)
    assert consumer.consumer == mock_consumer_instance
    assert consumer.running is False


@pytest.mark.asyncio
@patch("streaming.kafka_consumer.Consumer")
async def test_consume_messages_success(mock_consumer):
    """
    Test that consume_messages processes messages successfully.
    """
    mock_consumer_instance = MagicMock()
    mock_consumer.return_value = mock_consumer_instance

    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = TEST_MESSAGE_VALUE.encode("utf-8")

    # Make poll return one message then None to stop the loop
    def poll_side_effect(timeout):
        yield mock_msg
        while True:
            yield None

    mock_consumer_instance.poll.side_effect = poll_side_effect(0.1)

    mock_callback = AsyncMock()
    consumer = MarketDataConsumer(TEST_BOOTSTRAP_SERVERS, TEST_GROUP_ID, TEST_TOPICS)

    # Start consume_messages as a task
    consume_task = asyncio.create_task(
        consumer.consume_messages(mock_callback, batch_size=1)
    )

    # Give the consumer a chance to run and process the message
    await asyncio.sleep(0.1)  # Give some time for consume_messages to run

    # Signal the consumer to stop
    consumer.stop()

    # Wait for the consume_messages task to truly finish
    await consume_task

    mock_consumer_instance.poll.assert_called()
    mock_callback.assert_called_once_with({"symbol": "AAPL", "price": 150.0})
    mock_consumer_instance.close.assert_called_once()


@pytest.mark.asyncio
@patch("streaming.kafka_consumer.Consumer")
async def test_consume_messages_kafka_error(mock_consumer):
    """
    Test that consume_messages handles Kafka errors (not EOF).
    """
    mock_consumer_instance = MagicMock()
    mock_consumer.return_value = mock_consumer_instance

    mock_msg = MagicMock()
    mock_msg.error.return_value = MagicMock()  # Simulate an error
    mock_msg.error.return_value.code.return_value = -1  # Non-EOF error
    mock_msg.error.return_value.__str__.return_value = "Test Kafka Error"

    def poll_side_effect(timeout):
        yield mock_msg
        while True:
            yield None

    mock_consumer_instance.poll.side_effect = poll_side_effect(0.1)

    mock_callback = AsyncMock()
    consumer = MarketDataConsumer(TEST_BOOTSTRAP_SERVERS, TEST_GROUP_ID, TEST_TOPICS)

    with patch("streaming.kafka_consumer.logger") as mock_logger:
        consume_task = asyncio.create_task(
            consumer.consume_messages(mock_callback, batch_size=1)
        )
        await asyncio.sleep(0.1)  # Give some time for consume_messages to run
        consumer.stop()
        await consume_task

    mock_consumer_instance.poll.assert_called()
    mock_callback.assert_not_called()
    mock_logger.error.assert_called_once_with(
        "kafka_consumer_error", error="Test Kafka Error"
    )
    mock_consumer_instance.close.assert_called_once()


@pytest.mark.asyncio
@patch("streaming.kafka_consumer.Consumer")
async def test_stop_consumer(mock_consumer):
    """
    Test that the stop method sets running to False and closes the consumer.
    """
    mock_consumer_instance = MagicMock()
    mock_consumer.return_value = mock_consumer_instance
    mock_consumer_instance.poll.return_value = (
        None  # Ensure poll doesn't block indefinitely
    )

    consumer = MarketDataConsumer(TEST_BOOTSTRAP_SERVERS, TEST_GROUP_ID, TEST_TOPICS)

    # Start the consumer loop in a task
    consume_task = asyncio.create_task(consumer.consume_messages(AsyncMock()))
    await asyncio.sleep(
        0.1
    )  # Give some time for consume_messages to run # Let it run for a bit

    consumer.stop()
    await consume_task  # Await its completion after stop is called

    assert consumer.running is False
    mock_consumer_instance.close.assert_called_once()
