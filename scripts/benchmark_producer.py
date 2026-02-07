import asyncio
import os
import random
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from streaming.producer import MarketDataProducer


async def run_benchmark(num_messages=1000):
    producer = MarketDataProducer()

    print(f"Starting producer benchmark with {num_messages} messages...")

    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    start_time = time.time()

    for i in range(num_messages):
        symbol = random.choice(symbols)
        market_data = {
            "symbol": symbol,
            "timestamp": int(datetime.now().timestamp() * 1000000),
            "bid": 100.0 + random.uniform(-1, 1),
            "ask": 101.0 + random.uniform(-1, 1),
            "last": 100.5 + random.uniform(-1, 1),
            "volume": random.randint(100, 10000),
            "open_interest": random.randint(1000, 50000),
            "implied_volatility": 0.2 + random.uniform(0, 0.1),
            "delta": random.uniform(0, 1),
            "gamma": random.uniform(0, 0.1),
            "vega": random.uniform(0, 0.5),
            "theta": random.uniform(-0.1, 0),
        }

        await producer.produce_market_data("market-data", market_data, key=symbol)

        if (i + 1) % 100 == 0:
            print(f"Sent {i + 1} messages...")

    producer.flush()
    end_time = time.time()

    duration = end_time - start_time
    throughput = num_messages / duration

    print("\nBenchmark Complete!")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Throughput: {throughput:.2f} messages/second")


def test_schema_enforcement():
    print("\nVerifying schema enforcement...")
    producer = MarketDataProducer()

    # Missing required field 'bid'
    invalid_data = {
        "symbol": "AAPL",
        "timestamp": int(datetime.now().timestamp() * 1000000),
        # "bid": 150.0,
        "ask": 151.0,
        "last": 150.5,
        "volume": 1000,
        "open_interest": 500,
        "implied_volatility": 0.25,
        "delta": 0.5,
        "gamma": 0.05,
        "vega": 0.1,
        "theta": -0.02,
    }

    try:
        # This is synchronous call internally in AvroSerializer
        producer.avro_serializer(invalid_data, None)
        print("FAIL: Invalid data passed schema validation!")
        return False
    except Exception as e:
        print(f"SUCCESS: Schema enforcement caught invalid data: {e}")
        return True


if __name__ == "__main__":
    # Note: Requires Kafka and Schema Registry to be running for real verification
    # But we can still verify the serializer logic offline if needed

    if len(sys.argv) > 1 and sys.argv[1] == "--schema-only":
        test_schema_enforcement()
    else:
        # This will fail if Kafka is not running
        try:
            asyncio.run(run_benchmark(100))
            test_schema_enforcement()
        except Exception as e:
            print(f"Benchmark failed (Kafka likely not running): {e}")
            print("Try running with --schema-only to just verify serialization logic.")
            test_schema_enforcement()
