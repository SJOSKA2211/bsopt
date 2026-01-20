import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call, PropertyMock
import asyncio
import os
import inspect

from src.tasks.email_tasks import (
    send_transactional_email,
    send_batch_marketing_emails,
    email_service as email_service_instance,
)
from src.utils.cache import RateLimitTier

# Mock the module-level logger
@pytest.fixture(autouse=True)
def mock_email_tasks_logger():
    with patch('src.tasks.email_tasks.logger') as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_email_service_dependencies():
    with patch('src.tasks.email_tasks.os.getenv') as mock_getenv, \
         patch('src.tasks.email_tasks.TransactionalEmailService') as MockTransactionalEmailService, \
         patch('src.tasks.email_tasks.email_service', new_callable=MagicMock) as mock_module_email_service, \
         patch('src.tasks.email_tasks.rate_limiter') as mock_rate_limiter_module:
        
        # Configure os.getenv mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "EMAIL_SERVICE_API_KEY": "test_api_key",
            "FROM_EMAIL": "test@example.com",
        }.get(key, default)

        # Configure the module-level email_service mock
        mock_module_email_service.send_single_email.return_value = True
        mock_module_email_service.send_batch_emails.return_value = True
        
        # Also mock jinja_env for the module-level email_service
        mock_jinja_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Mock Email Content</html>"
        mock_jinja_env.get_template.return_value = mock_template
        type(mock_module_email_service).jinja_env = PropertyMock(return_value=mock_jinja_env) # Mock the property

        # Configure rate_limiter mock
        mock_rate_limiter_module.check_rate_limit = AsyncMock(return_value=True)
        
        yield (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
               mock_rate_limiter_module)






# --- Tests for send_transactional_email ---
def test_send_transactional_email_success(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies
    
    to_email = "to@example.com"
    subject = "Test Subject"
    template_name = "test_template.html"
    context = {"name": "Test User"}

    result = send_transactional_email.run(to_email, subject, template_name, context)
    
    mock_rate_limiter_module.check_rate_limit.assert_awaited_once_with(
        user_id="system_email", endpoint="send_email", tier=RateLimitTier.ENTERPRISE
    )
    mock_module_email_service.send_single_email.assert_called_once_with(
        to_email=to_email, subject=subject, template_name=template_name, context=context
    )
    mock_email_tasks_logger.info.assert_any_call(f"Sending email to {to_email}")
    mock_email_tasks_logger.warning.assert_not_called()
    assert result == {"status": "sent", "to": to_email}

def test_send_transactional_email_rate_limit_exceeded(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies
    
    mock_rate_limiter_module.check_rate_limit.return_value = False
    
    # When bind=True is removed, retry is not directly on 'self'.
    # We need to patch the task's actual retry method if it's still being called.
    # For now, let's just make sure the task itself can raise the exception for retry.
    with patch('src.tasks.email_tasks.send_transactional_email.retry') as mock_retry:
        mock_retry.side_effect = Exception("Celery retry triggered") # Simulate retry
        with pytest.raises(Exception, match="Celery retry triggered"):
            send_transactional_email.run("to@example.com", "Subject", "template.html", {})
    
        mock_email_tasks_logger.warning.assert_called_once_with("Email rate limit reached. Retrying in 60s...")
        mock_retry.assert_called_once_with(countdown=60)

def test_send_transactional_email_sendgrid_failure(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies
    
    mock_module_email_service.send_single_email.return_value = False
    
    with patch('src.tasks.email_tasks.send_transactional_email.retry') as mock_retry:
        mock_retry.side_effect = Exception("Celery retry triggered") # Simulate retry
        with pytest.raises(Exception, match="Celery retry triggered"):
            send_transactional_email.run("to@example.com", "Subject", "template.html", {})
    
        mock_retry.assert_called_once_with(exc=Exception("Failed to send email via SendGrid"))
def test_send_transactional_email_asyncio_loop_running(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies

    # Simulate a running event loop
    with patch('src.tasks.email_tasks.asyncio.get_event_loop') as mock_get_event_loop, \
         patch('concurrent.futures.ThreadPoolExecutor') as MockThreadPoolExecutor, \
         patch('src.tasks.email_tasks.asyncio.run') as mock_asyncio_run, \
         patch('src.tasks.email_tasks.send_transactional_email.retry') as mock_retry: # Patch retry here as well
        
        mock_retry.side_effect = Exception("Celery retry triggered") # Configure mock_retry

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mock_get_event_loop.return_value = mock_loop

        mock_executor = MagicMock()
        MockThreadPoolExecutor.return_value.__enter__.return_value = mock_executor
        mock_future = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_future.result.return_value = True # Rate limit check passes

        result = send_transactional_email.run("to@example.com", "Subject", "template.html", {})
        
        mock_get_event_loop.assert_called_once()
        mock_loop.is_running.assert_called_once()
        MockThreadPoolExecutor.assert_called_once()
        mock_executor.submit.assert_called_once()
        mock_future.result.assert_called_once()
        mock_asyncio_run.assert_called_once() # Called by the executor's lambda
        assert result == {"status": "sent", "to": "to@example.com"}
def test_send_transactional_email_runtime_error_no_loop(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies

    # Simulate RuntimeError from asyncio.get_event_loop()
    with patch('src.tasks.email_tasks.asyncio.get_event_loop') as mock_get_event_loop, \
         patch('src.tasks.email_tasks.asyncio.run') as mock_asyncio_run, \
         patch('src.tasks.email_tasks.send_transactional_email.retry') as mock_retry: # Patch retry here as well
        
        mock_retry.side_effect = Exception("Celery retry triggered") # Configure mock_retry

        mock_asyncio_run.return_value = True # Rate limit check passes
        
        result = send_transactional_email.run("to@example.com", "Subject", "template.html", {})
        
        mock_get_event_loop.assert_called_once()
        mock_asyncio_run.assert_called_once()
        assert result == {"status": "sent", "to": "to@example.com"}

# --- Tests for send_batch_marketing_emails ---
def test_send_batch_marketing_emails_success(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies
    
    recipients = [{"email": f"user{i}@example.com"} for i in range(150)]
    subject = "Batch Subject"
    template_name = "batch_template.html"

    result = send_batch_marketing_emails(recipients, subject, template_name)
    
    assert mock_module_email_service.send_batch_emails.call_count == 2 # 150 recipients / 100 BATCH_SIZE = 2 calls
    mock_module_email_service.send_batch_emails.assert_any_call(
        recipients[0:100], subject, template_name
    )
    mock_module_email_service.send_batch_emails.assert_any_call(
        recipients[100:150], subject, template_name
    )
    mock_email_tasks_logger.info.assert_any_call("Sending batch email to 150 recipients")
    assert result == {"status": "batch_sent", "count": 150}
def test_send_batch_marketing_emails_empty_recipients(mock_email_service_dependencies, mock_email_tasks_logger):
    (mock_getenv, MockTransactionalEmailService, mock_module_email_service, 
     mock_rate_limiter_module) = mock_email_service_dependencies
    
    recipients = []
    subject = "Batch Subject"
    template_name = "batch_template.html"

    result = send_batch_marketing_emails(recipients, subject, template_name)
    
    mock_module_email_service.send_batch_emails.assert_not_called()
    assert result == {"status": "batch_sent", "count": 0}
