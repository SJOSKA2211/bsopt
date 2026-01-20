import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
import asyncio
import json

# Import the FastAPI app directly for testing
from src.streaming.main import app, websocket_marketdata

@pytest.fixture(scope="module")
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    # async_client is already the yielded TestClient instance due to how pytest-asyncio handles async fixtures
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_websocket_marketdata_connect_disconnect(async_client: AsyncClient):
    async with async_client.websocket_connect("/marketdata") as websocket:
        await websocket.send_text("test") # Send a message to keep it alive briefly
        await websocket.close()
    # No explicit assertion needed, just checking for no unhandled exceptions
    # The client disconnection is handled in the websocket_marketdata function

@pytest.mark.asyncio
async def test_websocket_marketdata_sends_data(async_client: AsyncClient):
    with patch('src.streaming.main.asyncio.sleep', new=AsyncMock()) as mock_sleep, \
         patch('src.streaming.main.time.time', return_value=12345.0) as mock_time:
        
        async with async_client.websocket_connect("/marketdata") as websocket:
            # First send: consumer.running is True
            # Simulate receiving some data (to prevent endless loop in test)
            # await websocket.send_text("subscribe") 
            
            # Allow one iteration of the loop
            await asyncio.sleep(0.01) # Small sleep to allow websocket.send_text to be called
            
            # The websocket_marketdata sends data in its loop
            received_data_str = await websocket.receive_text()
            received_data = json.loads(received_data_str)

            assert "symbol" in received_data
            assert received_data["symbol"] == "AAPL"
            assert received_data["time"] == 12345
            assert mock_sleep.called
            
            # Force close to exit the loop
            await websocket.close()