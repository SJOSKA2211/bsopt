import pytest
from unittest.mock import MagicMock, patch, call, AsyncMock
import logging
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, Mail, Personalization

from src.services.email_service import (
    TransactionalEmailService,
    EMAILS_SENT_TOTAL,
    EMAIL_DELIVERY_TIME,
)

@pytest.fixture
def mock_email_service_dependencies():
    with patch('src.services.email_service.SendGridAPIClient') as MockSendGridAPIClient, \
         patch('src.services.email_service.Environment') as MockJinjaEnvironment, \
         patch('src.services.email_service.FileSystemLoader') as MockFileSystemLoader, \
         patch('src.services.email_service.select_autoescape') as MockSelectAutoescape, \
         patch('src.services.email_service.os.getcwd', return_value="/app"), \
         patch('src.services.email_service.os.makedirs') as MockMakedirs, \
         patch('src.services.email_service.logger') as mock_logger, \
         patch.object(EMAILS_SENT_TOTAL, 'labels', return_value=MagicMock()) as mock_emails_sent_labels, \
         patch.object(EMAIL_DELIVERY_TIME, 'labels', return_value=MagicMock()) as mock_email_delivery_time_labels:
        
        mock_sendgrid_client_instance = MockSendGridAPIClient.return_value
        
        mock_jinja_env_instance = MockJinjaEnvironment.return_value
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>rendered</html>"
        mock_jinja_env_instance.get_template.return_value = mock_template

        # Mock the time() method on the Histogram.labels object
        mock_email_delivery_time_labels.return_value.time.return_value.__enter__.return_value = None
        
        yield (MockSendGridAPIClient, mock_sendgrid_client_instance, 
               MockJinjaEnvironment, mock_jinja_env_instance, mock_template, 
               MockFileSystemLoader, MockSelectAutoescape, MockMakedirs, 
               mock_logger, mock_emails_sent_labels, mock_email_delivery_time_labels)

# --- Initialization Tests ---
def test_init_success(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     MockJinjaEnvironment, mock_jinja_env_instance, mock_template, 
     MockFileSystemLoader, MockSelectAutoescape, MockMakedirs, 
     mock_logger, _, _) = mock_email_service_dependencies

    service = TransactionalEmailService("test_api_key", "from@example.com")

    MockSendGridAPIClient.assert_called_once_with("test_api_key")
    assert service.client is mock_sendgrid_client_instance
    assert service.from_email == "from@example.com"
    
    expected_template_dir = os.path.join(os.getcwd(), "src", "templates", "emails")
    MockFileSystemLoader.assert_called_once_with(expected_template_dir)
    MockSelectAutoescape.assert_called_once_with(["html", "xml"])
    MockJinjaEnvironment.assert_called_once_with(
        loader=MockFileSystemLoader.return_value, autoescape=MockSelectAutoescape.return_value
    )
    MockMakedirs.assert_called_once_with(expected_template_dir, exist_ok=True)

# --- Template Rendering Tests ---
def test_render_template(mock_email_service_dependencies):
    (_, _, _, mock_jinja_env_instance, mock_template, _, _, _, _, _, _) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    template_name = "welcome.html"
    context = {"user": "Test User"}
    
    result = service._render_template(template_name, context)
    
    mock_jinja_env_instance.get_template.assert_called_once_with(template_name)
    mock_template.render.assert_called_once_with(user="Test User")
    assert result == "<html>rendered</html>"

# --- Single Email Tests ---
def test_send_single_email_success(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, mock_jinja_env_instance, mock_template, _, _, _, 
     mock_logger, mock_emails_sent_labels, mock_email_delivery_time_labels) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_sendgrid_client_instance.send.return_value = mock_response
    
    result = service.send_single_email("to@example.com", "Subject", "template.html", {"key": "value"})
    
    assert result is True
    mock_template.render.assert_called_once_with(key="value")
    mock_sendgrid_client_instance.send.assert_called_once()
    mock_emails_sent_labels.assert_called_with(status="success", type="transactional")
    mock_emails_sent_labels.return_value.inc.assert_called_once()
    mock_email_delivery_time_labels.assert_called_once_with(type="transactional")
    mock_email_delivery_time_labels.return_value.time.assert_called_once()
    mock_logger.error.assert_not_called()

def test_send_single_email_sendgrid_error(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, mock_jinja_env_instance, mock_template, _, _, _, 
     mock_logger, mock_emails_sent_labels, mock_email_delivery_time_labels) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.body = b"Bad Request"
    mock_sendgrid_client_instance.send.return_value = mock_response
    
    result = service.send_single_email("to@example.com", "Subject", "template.html", {"key": "value"})
    
    assert result is False
    # The actual logging call includes the f-string evaluated
    mock_logger.error.assert_called_once_with(f"SendGrid error: {mock_response.status_code} - {mock_response.body}")
    mock_emails_sent_labels.assert_called_with(status="error", type="transactional")
    mock_emails_sent_labels.return_value.inc.assert_called_once()

def test_send_single_email_exception(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, mock_jinja_env_instance, mock_template, _, _, _, 
     mock_logger, mock_emails_sent_labels, mock_email_delivery_time_labels) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    mock_sendgrid_client_instance.send.side_effect = Exception("Network issue")
    
    with pytest.raises(Exception, match="Network issue"):
        service.send_single_email("to@example.com", "Subject", "template.html", {"key": "value"})
    
    mock_logger.error.assert_called_once_with("Failed to send email to to@example.com: Network issue")
    mock_emails_sent_labels.assert_called_with(status="failed", type="transactional")
    mock_emails_sent_labels.return_value.inc.assert_called_once()

# --- Batch Email Tests ---
def test_send_batch_emails_success(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, mock_jinja_env_instance, mock_template, _, _, _, 
     mock_logger, _, _) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_sendgrid_client_instance.send.return_value = mock_response
    
    recipients = [
        {"email": "to1@example.com", "context": {"name": "User1"}},
        {"email": "to2@example.com", "context": {"name": "User2"}},
    ]
    subject = "Batch Subject"
    template_name = "batch.html"
    
    result = service.send_batch_emails(recipients, subject, template_name)
    
    assert result is True
    assert mock_jinja_env_instance.get_template.call_count == len(recipients)
    assert mock_template.render.call_count == len(recipients)
    mock_sendgrid_client_instance.send.assert_called_once()
    mock_logger.error.assert_not_called()

def test_send_batch_emails_empty_recipients(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, _, _, _, _, _, mock_logger, _, _) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    result = service.send_batch_emails([], "Batch Subject", "template.html")
    
    assert result is True
    mock_sendgrid_client_instance.send.assert_not_called()
    mock_logger.error.assert_not_called()

def test_send_batch_emails_exception(mock_email_service_dependencies):
    (MockSendGridAPIClient, mock_sendgrid_client_instance, 
     _, mock_jinja_env_instance, mock_template, _, _, _, 
     mock_logger, _, _) = mock_email_service_dependencies
    
    service = TransactionalEmailService("test_api_key", "from@example.com")
    
    mock_sendgrid_client_instance.send.side_effect = Exception("Batch send network error")
    
    recipients = [
        {"email": "to1@example.com", "context": {"name": "User1"}},
    ]
    
    result = service.send_batch_emails(recipients, "Batch Subject", "template.html")
    
    assert result is False
    mock_logger.error.assert_called_once_with(f"Batch send failed: Batch send network error")
