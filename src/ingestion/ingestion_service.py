import asyncio
import json
import os

import grpc

try:
    import Manifold_core
except ImportError:
    Manifold_core = None
import structlog

from src.shared.protos import market_data_pb2, market_data_pb2_grpc
from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)


class DataIngestionServicer(market_data_pb2_grpc.DataServiceServicer):
    def __init__(self):
        self.rmq = get_rabbitmq()
        self.last_price_cache = {}
        self.quarantine_path = "/tmp/ingestion_quarantine.bin"

        # Initialize High-Throughput Native Engine
        try:
            if Manifold_core:
                self.native_ingest = Manifold_core.PyNativeIngest(
                    "0.0.0.0:5555",  # High-speed UDP port
                    self.quarantine_path,
                )
                self.native_ingest.start()
                logger.info("native_ingest_engine_started", addr="0.0.0.0:5555")
            else:
                self.native_ingest = None
                logger.warning("manifold_core_not_found_native_ingest_disabled")
        except Exception as e:
            logger.error("native_ingest_start_failed", error=str(e))
            self.native_ingest = None

    async def IngestTicks(self, request, context):
        """
        Receives a batch of ticks, validates them via Rust, and pushes to RabbitMQ.
        """
        try:
            n = len(request.ticks)
            if n == 0:
                return market_data_pb2.IngestResponse(processed_count=0)

            # 1. Prepare data for Batch Validation
            prices = [float(tick.price) for tick in request.ticks]
            last_prices = [self.last_price_cache.get(tick.ticker, 0.0) for tick in request.ticks]

            # 2. High-speed Native Batch Validation
            valid_mask = [True] * n
            if Manifold_core:
                import numpy as np

                p_arr = np.array(prices, dtype=np.float64)
                lp_arr = np.array(last_prices, dtype=np.float64)
                valid_mask = Manifold_core.batch_validate_ticks(p_arr, lp_arr)

            batch = []
            for i, tick in enumerate(request.ticks):
                if not valid_mask[i]:
                    logger.warning(
                        "ingestion_tick_rejected_outlier", ticker=tick.ticker, price=tick.price
                    )
                    continue

                price = prices[i]
                ticker = tick.ticker
                self.last_price_cache[ticker] = price

                batch.append(
                    {
                        "time": float(tick.timestamp),
                        "symbol": ticker,
                        "last": price,
                        "source": tick.source,
                    }
                )

            # 3. Optimized Asynchronous Publishing
            if batch:
                # Fire and forget if needed or await for strict consistency
                await self.rmq.publish_batch(batch)
                logger.info(
                    "ingestion_batch_processed",
                    count=len(batch),
                    rejected=n - len(batch),
                )

            return market_data_pb2.IngestResponse(processed_count=len(batch))
        except Exception as e:
            logger.error("ingestion_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return market_data_pb2.IngestResponse(processed_count=0)

    async def GetHistoricalData(self, request, context):
        """
        Retrieves historical data from TimescaleDB.
        """
        try:
            from src.database.pipeliner import db_engine

            ticker = request.ticker or settings.DEFAULT_TICKER
            # In a real scenario, we would use start_time and end_time from request
            records = await db_engine.fetch_training_data([ticker], limit=1000)

            ticks = []
            for r in records:
                # Convert datetime to timestamp if necessary
                ts = (
                    int(r["time"].timestamp())
                    if hasattr(r["time"], "timestamp")
                    else int(r["time"])
                )
                ticks.append(
                    market_data_pb2.Tick(
                        ticker=r["symbol"],
                        price=float(r["last"]),
                        timestamp=ts,
                        source="timescaledb",
                    )
                )

            return market_data_pb2.HistoryResponse(data=ticks)
        except Exception as e:
            logger.error("historical_data_fetch_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return market_data_pb2.HistoryResponse()


async def serve():
    server = grpc.aio.server()
    servicer = DataIngestionServicer()
    market_data_pb2_grpc.add_DataServiceServicer_to_server(servicer, server)
    listen_addr = "[::]:50053"
    server.add_insecure_port(listen_addr)
    logger.info("ingestion_grpc_server_started", addr=listen_addr)
    await server.start()

    # Heartbeat task
    async def heartbeat():
        import time

        while True:
            try:
                metrics = {}
                if servicer.native_ingest:
                    metrics_str = servicer.native_ingest.get_metrics()
                    metrics = json.loads(metrics_str)

                with open("/tmp/ingestion_heartbeat", "w") as f:
                    heartbeat_data = {"time": time.time(), "metrics": metrics}
                    f.write(json.dumps(heartbeat_data))
                    # Ensure the file is flushed
                    os.fsync(f.fileno()) if hasattr(os, "fsync") else None
            except Exception as e:
                logger.error("heartbeat_failed", error=str(e))
            await asyncio.sleep(5)  # Frequent heartbeats for high-throughput monitoring

    asyncio.create_task(heartbeat())
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
