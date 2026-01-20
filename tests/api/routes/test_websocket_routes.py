import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.websockets.manager import manager, ConnectionMetadata
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.websockets import WebSocketDisconnect
import asyncio

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_ws_manager():
    # Patch the singleton instance's methods directly
    # This ensures that ANY module using this instance will see the mocks
    original_connect = manager.connect
    original_disconnect = manager.disconnect
    
    async def fake_connect(websocket, symbol):
        await websocket.accept()
        if not hasattr(websocket, "metadata"):
            websocket.metadata = ConnectionMetadata()
            
    manager.connect = AsyncMock(side_effect=fake_connect)
    manager.disconnect = MagicMock()
    
    yield manager.connect, manager.disconnect
    
    # Restore original methods
    manager.connect = original_connect
    manager.disconnect = original_disconnect

def test_market_data_ws_connection(mock_ws_manager):
    m_conn, m_disc = mock_ws_manager
    symbol = "AAPL"
    
    with client.websocket_connect(f"/ws/market-data?symbol={symbol}") as websocket:
        websocket.send_text("hello")
        # Loop will wait for next message.
        # Closing the context will trigger disconnect on server side.
    
    m_conn.assert_called()
    m_disc.assert_called()

def test_market_data_ws_protocols(mock_ws_manager):
    for proto in ["json", "proto", "msgpack"]:
        with client.websocket_connect(f"/ws/market-data?symbol=MSFT&protocol={proto}") as websocket:
            websocket.send_text("ping")

def test_market_data_ws_exception_handling(mock_ws_manager):
    m_conn, m_disc = mock_ws_manager
    # Patch receive_text on the WebSocket instance that will be created
    with patch("src.api.routes.websocket.WebSocket.receive_text", side_effect=Exception("Boom")):
        try:
            with client.websocket_connect("/ws/market-data?symbol=AAPL") as websocket:
                websocket.send_text("trigger")
        except:
            pass
    
    # Wait a tiny bit for the server coroutine to process
    import time
    time.sleep(0.1)
    m_disc.assert_called()