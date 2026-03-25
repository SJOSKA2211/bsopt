from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.mesh_publisher import MarketMeshPublisher


@pytest.mark.asyncio
async def test_mesh_publisher_success():
    """Verify MarketMeshPublisher correctly formats and publishes ticks to RabbitMQ."""
    with patch("src.ingestion.mesh_publisher.get_rabbitmq") as mock_get_rmq:
        mock_rmq = AsyncMock()
        mock_get_rmq.return_value = mock_rmq
        
        publisher = MarketMeshPublisher()
        test_data = {
            "AAPL": {"price": 150.0, "volume": 1000, "time": 123456789.0, "side": 1}
        }
        
        await publisher.publish(test_data)
        
        mock_rmq.publish_tick.assert_called_once_with({
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000,
            "time": 123456789.0,
            "side": 1
        })

@pytest.mark.asyncio
async def test_mesh_publisher_error_handling():
    """Verify MarketMeshPublisher logs error but doesn't crash on failure."""
    with patch("src.ingestion.mesh_publisher.get_rabbitmq") as mock_get_rmq:
        mock_rmq = AsyncMock()
        mock_rmq.publish_tick.side_effect = Exception("RMQ Down")
        mock_get_rmq.return_value = mock_rmq
        
        publisher = MarketMeshPublisher()
        await publisher.publish({"ERR": {"price": 10.0}}) # Should not raise
