import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.workers.webhook_worker import process_webhook_task, send_to_dlq_task, _process_webhook_core
import asyncio
from celery.exceptions import MaxRetriesExceededError

@pytest.fixture
def mock_dispatcher():
    with patch("src.workers.webhook_worker.get_webhook_dispatcher") as mock_get:
        mock_disp = AsyncMock()
        mock_get.return_value = mock_disp
        yield mock_disp

@pytest.mark.asyncio
async def test_process_webhook_core_success(mock_dispatcher):
    mock_self = MagicMock()
    mock_self.request.retries = 0
    webhook_data = {
        "url": "http://example.com/hook",
        "payload": {"event": "test"},
        "headers": {},
        "secret": "secret"
    }
    
    await _process_webhook_core(mock_self, webhook_data)
    mock_dispatcher.dispatch_webhook.assert_called_once()

@pytest.mark.asyncio
async def test_process_webhook_core_retry(mock_dispatcher):
    mock_self = MagicMock()
    mock_self.request.retries = 0
    mock_dispatcher.dispatch_webhook.side_effect = Exception("Dispatch failed")
    
    webhook_data = {
        "url": "http://example.com/hook",
        "payload": {},
        "headers": {},
        "secret": "s"
    }
    
    await _process_webhook_core(mock_self, webhook_data)
    mock_self.retry.assert_called_once()

@pytest.mark.asyncio
async def test_process_webhook_core_max_retries(mock_dispatcher):
    mock_self = MagicMock()
    mock_self.request.retries = 5
    mock_dispatcher.dispatch_webhook.side_effect = Exception("Permanent failure")
    mock_self.retry.side_effect = MaxRetriesExceededError()
    
    webhook_data = {"url": "http://ex.com", "payload": {}, "headers": {}, "secret": "s"}
    
    with patch("src.workers.webhook_worker.send_to_dlq_task") as mock_dlq:
        await _process_webhook_core(mock_self, webhook_data)
        mock_dlq.delay.assert_called_once()

def test_send_to_dlq_task():
    with patch("src.workers.webhook_worker.logger") as mock_logger:
        send_to_dlq_task({"url": "http://test.com"}, "test_reason")
        mock_logger.error.assert_called()

def test_process_webhook_task_entrypoint():
    # Since it uses asyncio.run, we mock the core function
    mock_self = MagicMock()
    webhook_data = {"some": "data"}
    with patch("src.workers.webhook_worker._process_webhook_core") as mock_core:
        # recalibrate_symbol showed us that bound tasks in this project 
        # seem to behave as methods.
        # Let's try calling it without explicit self first.
        try:
            process_webhook_task(webhook_data)
        except TypeError:
            # If it's NOT bound or behaving differently
            process_webhook_task.__wrapped__(mock_self, webhook_data)
        
        mock_core.assert_called_once()
