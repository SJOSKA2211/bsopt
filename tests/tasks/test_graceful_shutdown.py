import pytest
import asyncio
import signal
from unittest.mock import MagicMock, AsyncMock, patch
from src.tasks.graceful_shutdown import ShutdownManager

@pytest.mark.asyncio
async def test_shutdown_manager_register_and_shutdown():
    manager = ShutdownManager()
    task1 = AsyncMock()
    task2 = AsyncMock()
    
    manager.register_cleanup(task1)
    manager.register_cleanup(task2)
    
    # We need to mock loop.stop() and asyncio.all_tasks() to prevent the test loop from stopping
    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        with patch("asyncio.all_tasks", return_value=[]):
            await manager.shutdown()
            
            task1.assert_awaited_once()
            task2.assert_awaited_once()
            mock_loop.stop.assert_called_once()

@pytest.mark.asyncio
async def test_shutdown_manager_error_in_task():
    manager = ShutdownManager()
    task1 = AsyncMock(side_effect=Exception("Test error"))
    task2 = AsyncMock()
    
    manager.register_cleanup(task1)
    manager.register_cleanup(task2)
    
    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        with patch("asyncio.all_tasks", return_value=[]):
            await manager.shutdown()
            
            task1.assert_awaited_once()
            task2.assert_awaited_once()
            mock_loop.stop.assert_called_once()

@pytest.mark.asyncio
async def test_shutdown_manager_with_signal():
    manager = ShutdownManager()
    mock_sig = MagicMock()
    mock_sig.name = "SIGTERM"
    
    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        with patch("asyncio.all_tasks", return_value=[]):
            await manager.shutdown(sig=mock_sig)
            mock_loop.stop.assert_called_once()

def test_install_signal_handlers():
    manager = ShutdownManager()
    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        manager.install_signal_handlers()
        
        assert mock_loop.add_signal_handler.call_count == 2
        # Check that it called with SIGINT and SIGTERM
        args_list = [call[0][0] for call in mock_loop.add_signal_handler.call_args_list]
        assert signal.SIGINT in args_list
        assert signal.SIGTERM in args_list
