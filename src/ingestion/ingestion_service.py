import asyncio

import grpc
import structlog

try:
    import Manifold_core
except ImportError:
    Manifold_core = None

from src.shared.protos import data_pb2, data_pb2_grpc
from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)

class DataIngestionServicer(data_pb2_grpc.DataServiceServicer):
    """
    gRPC Servicer for Centralized Data Ingestion.
    Receives ticks from scrapers and publishes to RabbitMQ.
    """

    def __init__(self):
        # Initialize RabbitMQ Client
        self.rmq = get_rabbitmq()
        # In-memory cache for outlier detection
        self.last_price_cache = {}

    async def IngestTicks(self, request, context):
        """
        Receives a batch of ticks, validates them via Rust, and pushes to RabbitMQ.
        """
        try:
            batch = []
            for tick in request.ticks:
                price = float(tick.price)
                ticker = tick.ticker

                # 1. High-speed Rust Validation
                is_valid = True
                if Manifold_core:
                    last_price = self.last_price_cache.get(ticker, 0.0)
                    is_valid = Manifold_core.validate_tick(ticker, price, last_price)

                if not is_valid:
                    logger.warning("ingestion_tick_rejected_outlier", ticker=ticker, price=price)
                    continue

                # Update cache
                self.last_price_cache[ticker] = price

                batch.append(
                    {
                        "time": float(tick.timestamp),
                        "symbol": ticker,
                        "last": price,
                        "source": tick.source,
                    }
                )

            # 2. Publish validated batch to RabbitMQ
            if batch:
                await self.rmq.publish_batch(batch)
                logger.info(
                    "ingestion_batch_processed",
                    count=len(batch),
                    rejected=len(request.ticks) - len(batch),
                )

            return data_pb2.IngestResponse(processed_count=len(batch))
        except Exception as e:
            logger.error("ingestion_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.IngestResponse(processed_count=0)

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
                    data_pb2.Tick(
                        ticker=r["symbol"],
                        price=float(r["last"]),
                        timestamp=ts,
                        source="timescaledb",
                    )
                )

            return data_pb2.HistoryResponse(data=ticks)
        except Exception as e:
            logger.error("historical_data_fetch_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_pb2.HistoryResponse()

async def serve():
    server = grpc.aio.server()
    data_pb2_grpc.add_DataServiceServicer_to_server(DataIngestionServicer(), server)
    listen_addr = "[::]:50053"
    server.add_insecure_port(listen_addr)
    logger.info("ingestion_grpc_server_started", addr=listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
