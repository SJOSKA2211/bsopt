from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from celery.exceptions import MaxRetriesExceededError

from src.shared.webhooks.dispatcher import WebhookDispatcher
from src.workers.webhook_worker import _process_webhook_core, send_to_dlq_task


@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock(spec=WebhookDispatcher)
    dispatcher.dispatch_webhook = AsyncMock()
    return dispatcher


@pytest.mark.asyncio
async def test_process_webhook_task_success(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret",
    }

    mock_task_self = MagicMock()
    mock_task_self.request.retries = 0
    mock_task_self.retry = MagicMock()

    with patch(
        "src.workers.webhook_worker.get_webhook_dispatcher",
        return_value=mock_dispatcher,
    ):
        await _process_webhook_core(mock_task_self, webhook_data)

        mock_dispatcher.dispatch_webhook.assert_called_once_with(
            url=webhook_data["url"],
            payload=webhook_data["payload"],
            headers=webhook_data["headers"],
            secret=webhook_data["secret"],
            retries=0,
        )


@pytest.mark.asyncio
async def test_process_webhook_task_failure_and_retry(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret",
    }

    mock_dispatcher.dispatch_webhook.side_effect = Exception("Simulated Dispatch Error")

    mock_task_self = MagicMock()
    mock_task_self.request.retries = 0
    mock_task_self.retry = MagicMock()

    with patch(
        "src.workers.webhook_worker.send_to_dlq_task.delay", new_callable=MagicMock
    ) as mock_dlq_task_delay:
        with patch(
            "src.workers.webhook_worker.get_webhook_dispatcher",
            return_value=mock_dispatcher,
        ):
            await _process_webhook_core(mock_task_self, webhook_data)

            mock_dispatcher.dispatch_webhook.assert_called_once_with(
                url=webhook_data["url"],
                payload=webhook_data["payload"],
                headers=webhook_data["headers"],
                secret=webhook_data["secret"],
                retries=0,
            )
            mock_task_self.retry.assert_called_once()
            mock_dlq_task_delay.assert_not_called()


@pytest.mark.asyncio
async def test_process_webhook_task_max_retries_exceeded(mock_dispatcher):
    webhook_data = {
        "url": "http://example.com/webhook",
        "payload": {"event": "test"},
        "headers": {"Content-Type": "application/json"},
        "secret": "test_secret",
    }

    mock_dispatcher.dispatch_webhook.side_effect = Exception("Simulated Dispatch Error")

    mock_task_self = MagicMock()
    mock_task_self.request.retries = 5
    mock_task_self.retry = MagicMock(side_effect=MaxRetriesExceededError("max retries"))

    with patch(
        "src.workers.webhook_worker.send_to_dlq_task.delay", new_callable=MagicMock
    ) as mock_dlq_task_delay:
        with patch(
            "src.workers.webhook_worker.get_webhook_dispatcher",
            return_value=mock_dispatcher,
        ):
            await _process_webhook_core(mock_task_self, webhook_data)

            mock_dispatcher.dispatch_webhook.assert_called_once_with(
                url=webhook_data["url"],
                payload=webhook_data["payload"],
                headers=webhook_data["headers"],
                secret=webhook_data["secret"],
                retries=5,
            )
            mock_task_self.retry.assert_called_once()
            mock_dlq_task_delay.assert_called_once()
            args, kwargs = mock_dlq_task_delay.call_args
            assert args[0]["url"] == webhook_data["url"]
            assert "max_retries" in kwargs["reason"]


@pytest.mark.asyncio
async def test_send_to_dlq_task_execution():
    webhook_data = {
        "url": "http://example.com/dlq",
        "payload": {"event": "dlq"},
        "headers": {},
        "secret": "dlq_secret",
        "reason": "max_retries_reached",
    }

    with patch("src.shared.utils.cache.get_redis_client", new_callable=AsyncMock):
        send_to_dlq_task(webhook_data, reason="test_dlq")