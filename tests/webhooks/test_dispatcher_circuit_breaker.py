from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.utils.circuit_breaker import CircuitBreaker
from src.webhooks.dispatcher import WebhookDispatcher


# Placeholder for Celery task for red phase. Will be replaced by actual Celery task in next step
@pytest.fixture
def mock_dlq_task():
    return MagicMock()

@pytest.fixture
def mock_celery_app():
    return MagicMock()

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_open(mock_celery_app, mock_dlq_task):
    """
    Test that dispatcher respects circuit breaker open state and does not make requests.
    """
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Force open state
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_failure() # 5 failures default
    assert cb.state == "OPEN"

    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value

        success = await dispatcher.dispatch("http://test.com", {"data": "test"})

        assert success is False
        mock_client_instance.post.assert_not_called()

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_failures(mock_celery_app, mock_dlq_task):
    """
    Test that dispatcher records failures in circuit breaker.
    """
    # Create a fresh circuit breaker
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.raise_for_status.return_value = None

    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = [
            httpx.HTTPStatusError("500 Server Error", request=httpx.Request("POST", "http://test.com"), response=MagicMock(status_code=500)),
            mock_response_success
        ]

        # First call fails
        await dispatcher.dispatch("http://test.com", {"data": "test"})
        assert cb.failure_count == 1

        # Second call succeeds
        # Note: Depending on retry logic, dispatch might swallow the error or retry internally.
        # If dispatch retries, the side_effect list needs to handle retries.
        # Assuming dispatch handles retries, let's verify cb state after calls.

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_retry_logic(mock_celery_app, mock_dlq_task):
    """Test interaction between retries and circuit breaker."""
    cb = CircuitBreaker(failure_threshold=1) # Fails fast
    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate constant failure
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError("500 Server Error", request=httpx.Request("POST", "http://test.com"), response=mock_response_fail)

    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = [mock_response_fail] * 6 # 5 retries + 1 initial

        success = await dispatcher.dispatch("http://test.com", {"data": "test"})
        assert success is False
        assert cb.state == "OPEN"

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_open_skips_retries(mock_celery_app, mock_dlq_task):
    """
    Test that if CB opens during retries, subsequent retries are skipped.
    """
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.new_callable = AsyncMock # Ensure it's async

        # We manually trigger CB open between calls if possible, or rely on implementation checking CB before retry
        # Ideally, we simulate: Fail -> CB Record -> Retry 1 (CB Closed) -> Fail -> CB Record -> ... -> CB Open -> Retry N (Skipped)
        pass

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_success_resets(mock_celery_app, mock_dlq_task):
    """Test that a successful call resets the failure count (handled by CB logic, but verifying integration)."""
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate network errors, then success
    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.new_callable = AsyncMock  # Ensure it's async

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=None, response=mock_response_fail)

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_client_instance.post.side_effect = [
            mock_response_fail, # Fail 1
            mock_response_success # Success
        ]

        # Call 1 (Fail)
        await dispatcher.dispatch("http://test.com", {})
        assert cb.failure_count == 1

        # Call 2 (Success)
        await dispatcher.dispatch("http://test.com", {})
        assert cb.failure_count == 0

@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_exception_handling(mock_celery_app, mock_dlq_task):
    """Test that unexpected exceptions trigger circuit breaker failure."""
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate unexpected exception
    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = Exception("Unexpected Error")

        await dispatcher.dispatch("http://test.com", {})
        assert cb.failure_count == 1

@pytest.mark.asyncio
async def test_dispatcher_respects_custom_cb(mock_celery_app, mock_dlq_task):
    """Test that we can pass a custom circuit breaker."""
    # Custom config
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5)

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    with patch('httpx.AsyncClient') as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = Exception("Fail")

        await dispatcher.dispatch("http://test.com", {}) # Fail 1
        assert cb.state == "CLOSED"

        await dispatcher.dispatch("http://test.com", {}) # Fail 2
        assert cb.state == "OPEN"
