from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from celery.exceptions import MaxRetriesExceededError  # Import the correct exception

from src.webhooks.dispatcher import WebhookDispatcher

# Import actual worker and dispatcher
from src.workers.webhook_worker import _process_webhook_core, send_to_dlq_task


@pytest.fixture
def mock_dispatcher():
    # Mock the WebhookDispatcher for isolation
    dispatcher = MagicMock(spec=WebhookDispatcher)
    dispatcher.dispatch_webhook = AsyncMock()
    return dispatcher

@pytest.mark.asyncio
async def test_process_webhook_task_success(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret"
    }
    
    # Create a dummy task instance to simulate Celery's 'self'
    mock_task_self = MagicMock()
    mock_task_self.request.retries = 0 # Simulate initial call
    mock_task_self.retry = MagicMock() # Mock the retry method
    
    # Mock get_webhook_dispatcher to return our mocked dispatcher
    with patch("src.workers.webhook_worker.get_webhook_dispatcher", return_value=mock_dispatcher):
        # Call the core function directly
        await _process_webhook_core(mock_task_self, webhook_data)
            
        mock_dispatcher.dispatch_webhook.assert_called_once_with(
            url=webhook_data["url"],
            payload=webhook_data["payload"],
            headers=webhook_data["headers"],
            secret=webhook_data["secret"],
            retries=0 # Initial call has 0 retries
        )

@pytest.mark.asyncio
async def test_process_webhook_task_failure_and_retry(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret",
    }
    
    # Simulate dispatcher failure -> Celery retry logic
    mock_dispatcher.dispatch_webhook.side_effect = Exception("Simulated Dispatch Error")
    
    # Create a dummy task instance to simulate Celery's 'self'
    mock_task_self = MagicMock()
    mock_task_self.request.retries = 0 # First retry attempt
    mock_task_self.retry = MagicMock() # Mock the retry method (without side_effect)
    
    # Patch the real send_to_dlq_task.delay in the worker module
    with patch("src.workers.webhook_worker.send_to_dlq_task.delay", new_callable=MagicMock) as mock_dlq_task_delay:
        with patch("src.workers.webhook_worker.get_webhook_dispatcher", return_value=mock_dispatcher):
            # Call the core function directly
            await _process_webhook_core(mock_task_self, webhook_data)
            
            mock_dispatcher.dispatch_webhook.assert_called_once_with(
                url=webhook_data["url"],
                payload=webhook_data["payload"],
                headers=webhook_data["headers"],
                secret=webhook_data["secret"],
                retries=0
            )
            mock_task_self.retry.assert_called_once() # Verify retry was called
            mock_dlq_task_delay.assert_not_called() # Should retry, not go to DLQ yet

@pytest.mark.asyncio
async def test_process_webhook_task_max_retries_exceeded(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret",
    }
    
    mock_dispatcher.dispatch_webhook.side_effect = Exception("Simulated Dispatch Error")
    
    # Create a dummy task instance to simulate Celery's 'self'
    mock_task_self = MagicMock()
    mock_task_self.request.retries = 5 # Max retries
    mock_task_self.retry = MagicMock(side_effect=MaxRetriesExceededError("max retries")) # Simulate retry failing
    
    with patch("src.workers.webhook_worker.send_to_dlq_task.delay", new_callable=MagicMock) as mock_dlq_task_delay:
        with patch("src.workers.webhook_worker.get_webhook_dispatcher", return_value=mock_dispatcher):
            await _process_webhook_core(mock_task_self, webhook_data)
            
            mock_dispatcher.dispatch_webhook.assert_called_once_with(
                url=webhook_data["url"],
                payload=webhook_data["payload"],
                headers=webhook_data["headers"],
                secret=webhook_data["secret"],
                retries=5
            )
            mock_task_self.retry.assert_called_once() # Verify retry was called
            mock_dlq_task_delay.assert_called_once()
            args, kwargs = mock_dlq_task_delay.call_args
            assert args[0]["url"] == webhook_data["url"]
            assert "celery_max_retries" in kwargs["reason"] # Access reason from kwargs

@pytest.mark.asyncio
async def test_send_to_dlq_task_execution():
    webhook_data = {
        "url": "http://example.com/dlq",
        "payload": {"event": "dlq"},
        "headers": {},
        "secret": "dlq_secret",
        "reason": "max_retries_reached"
    }
    
    # Just call the task directly and ensure it runs without error
    send_to_dlq_task(webhook_data, reason="test_dlq")
    # No explicit assertions needed, as success is no exception being raised.
    # In a real test, one might mock a persistent storage or log to verify.