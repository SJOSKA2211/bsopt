from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.ingestion_service import DataIngestionServicer
from src.shared.protos import market_data_pb2


@pytest.mark.asyncio
async def test_ingestion_service_to_rabbitmq():
    """
    Integration test for DataIngestionServicer -> RabbitMQ path.
    """
    # 1. Setup Servicer
    with patch("src.ingestion.ingestion_service.get_rabbitmq") as mock_get_rmq:
        mock_rmq = AsyncMock()
        mock_get_rmq.return_value = mock_rmq

        servicer = DataIngestionServicer()

        # 2. Mock Request
        request = market_data_pb2.IngestRequest(
            ticks=[
                market_data_pb2.Tick(
                    ticker="AAPL", price=150.0, timestamp=1700000000, source="test"
                ),
                market_data_pb2.Tick(
                    ticker="GOOGL", price=2800.0, timestamp=1700000001, source="test"
                ),
            ]
        )
        context = MagicMock()

        # 3. Execute IngestTicks
        # We need to mock Manifold_core if it exists, or just let it return True
        with patch("src.ingestion.ingestion_service.Manifold_core") as mock_core:
            mock_core.validate_tick.return_value = True

            response = await servicer.IngestTicks(request, context)

            # 4. Verify
            assert response.processed_count == 2
            mock_rmq.publish_batch.assert_called_once()

            # Check content of the batch
            batch = mock_rmq.publish_batch.call_args[0][0]
            assert len(batch) == 2
            assert batch[0]["symbol"] == "AAPL"
            assert batch[0]["last"] == 150.0
            assert batch[1]["symbol"] == "GOOGL"


@pytest.mark.asyncio
async def test_ingestion_service_outlier_rejection():
    """Test that outliers are rejected and not published to RabbitMQ."""
    with patch("src.ingestion.ingestion_service.get_rabbitmq") as mock_get_rmq:
        mock_rmq = AsyncMock()
        mock_get_rmq.return_value = mock_rmq

        servicer = DataIngestionServicer()

        # First tick to establish baseline
        request1 = market_data_pb2.IngestRequest(
            ticks=[
                market_data_pb2.Tick(
                    ticker="AAPL", price=100.0, timestamp=1700000000, source="test"
                )
            ]
        )
        context = MagicMock()

        with patch("src.ingestion.ingestion_service.Manifold_core") as mock_core:
            # First tick valid
            mock_core.validate_tick.return_value = True
            await servicer.IngestTicks(request1, context)

            # Second tick is an outlier
            request2 = market_data_pb2.IngestRequest(
                ticks=[
                    market_data_pb2.Tick(
                        ticker="AAPL", price=500.0, timestamp=1700000005, source="test"
                    )
                ]
            )
            mock_core.validate_tick.return_value = False  # Rejected by Rust core

            response = await servicer.IngestTicks(request2, context)

            assert response.processed_count == 0
            # publish_batch should only have been called for the first tick
            assert mock_rmq.publish_batch.call_count == 1
