import asyncio
import time

import pytest

from src.ingestion.xdp_ingest import XDPIngester
from src.workers.streaming.kafka_consumer import MarketDataConsumer


@pytest.mark.asyncio
async def test_xdp_ingester_throughput_benchmark():
    # Setup
    ingester = XDPIngester(interface="lo", port=9999)
    # Mock socket to avoid permission issues in CI
    ingester.sock = asyncio.create_task(asyncio.sleep(0.1))  # Dummy
    ingester._running = True

    # We want to measure decoding and mesh writing speed
    num_ticks = 10000
    tick_data = b'{"symbol": "AAPL", "price": 150.0, "volume": 100, "timestamp": 1644300000.0}'

    start_time = time.time()
    for _ in range(num_ticks):
        ingester._handle_packet(tick_data)
    duration = time.time() - start_time

    throughput = num_ticks / duration
    print(f"\nXDP Ingester Throughput: {throughput:.2f} ticks/sec")

    assert throughput > 50000


@pytest.mark.asyncio
async def test_kafka_consumer_batch_processing_speed():
    # Setup
    consumer = MarketDataConsumer()

    # Mock batch
    batch_size = 1000
    mock_batch = [
        {"symbol": "TSLA", "price": 200.0, "timestamp": "2026-02-08T08:00:00"}
        for _ in range(batch_size)
    ]

    # Mock callback
    async def mock_callback(msg):
        await asyncio.sleep(0.00001)  # Simulate minor processing

    start_time = time.time()
    await consumer._process_batch(mock_batch, mock_callback)
    duration = time.time() - start_time

    throughput = batch_size / duration
    print(f"Kafka Batch Throughput: {throughput:.2f} msgs/sec")

    assert throughput > 10000
