import pytest
from unittest.mock import MagicMock, patch
import logging
from src.tasks.audit_tasks import persist_audit_log

# Mock the module-level logger
@pytest.fixture(autouse=True)
def mock_audit_tasks_logger():
    with patch('src.tasks.audit_tasks.logger') as mock_logger:
        yield mock_logger

# Mock external dependencies for the task
@pytest.fixture
def mock_audit_task_dependencies():
    with patch('src.tasks.audit_tasks.get_session') as mock_get_session, \
         patch('src.tasks.audit_tasks.AuditLog') as MockAuditLog:
        
        mock_session_instance = MagicMock()
        mock_get_session.return_value = mock_session_instance
        
        yield mock_get_session, mock_session_instance, MockAuditLog

def test_persist_audit_log_success(mock_audit_task_dependencies, mock_audit_tasks_logger):
    mock_get_session, mock_session_instance, MockAuditLog = mock_audit_task_dependencies
    
    event_type = "LOGIN_SUCCESS"
    user_id = "user123"
    user_email = "masked@example.com"
    source_ip = "192.168.1.xxx"
    user_agent = "TestAgent"
    request_path = "/auth/login"
    request_method = "POST"
    details = {"login_method": "password"}

    persist_audit_log(
        event_type=event_type,
        user_id=user_id,
        user_email=user_email,
        source_ip=source_ip,
        user_agent=user_agent,
        request_path=request_path,
        request_method=request_method,
        details=details,
    )
    
    MockAuditLog.assert_called_once_with(
        event_type=event_type,
        user_id=user_id,
        user_email=user_email,
        source_ip=source_ip,
        user_agent=user_agent,
        request_path=request_path,
        request_method=request_method,
        details=details,
    )
    mock_session_instance.add.assert_called_once_with(MockAuditLog.return_value)
    mock_session_instance.commit.assert_called_once()
    mock_session_instance.close.assert_called_once()
    mock_session_instance.rollback.assert_not_called()
    mock_audit_tasks_logger.info.assert_called_once_with(f"Audit log persisted: {event_type} for user {user_id}")
    mock_audit_tasks_logger.error.assert_not_called()

def test_persist_audit_log_exception_rollback(mock_audit_task_dependencies, mock_audit_tasks_logger):
    mock_get_session, mock_session_instance, MockAuditLog = mock_audit_task_dependencies
    
    mock_session_instance.commit.side_effect = Exception("Database write error")

    event_type = "LOGIN_FAILURE"
    user_id = "user123"
    user_email = "masked@example.com"
    source_ip = "192.168.1.xxx"
    user_agent = "TestAgent"
    request_path = "/auth/login"
    request_method = "POST"
    details = {"reason": "bad_password"}

    persist_audit_log(
        event_type=event_type,
        user_id=user_id,
        user_email=user_email,
        source_ip=source_ip,
        user_agent=user_agent,
        request_path=request_path,
        request_method=request_method,
        details=details,
    )
    
    mock_session_instance.add.assert_called_once_with(MockAuditLog.return_value)
    mock_session_instance.commit.assert_called_once()
    mock_session_instance.rollback.assert_called_once()
    mock_session_instance.close.assert_called_once()
    mock_audit_tasks_logger.error.assert_called_once_with("Failed to persist audit log to database via Celery task: Database write error")
    mock_audit_tasks_logger.info.assert_not_called()

def test_persist_audit_log_exception_no_session(mock_audit_task_dependencies, mock_audit_tasks_logger):
    mock_get_session, mock_session_instance, MockAuditLog = mock_audit_task_dependencies
    
    mock_get_session.side_effect = Exception("Database connection error")

    event_type = "CRITICAL_ERROR"
    user_id = None
    user_email = None
    source_ip = None
    user_agent = None
    request_path = None
    request_method = None
    details = {"error": "DB init failure"}

    persist_audit_log(
        event_type=event_type,
        user_id=user_id,
        user_email=user_email,
        source_ip=source_ip,
        user_agent=user_agent,
        request_path=request_path,
        request_method=request_method,
        details=details,
    )
    
    MockAuditLog.assert_not_called()
    mock_session_instance.add.assert_not_called()
    mock_session_instance.commit.assert_not_called()
    mock_session_instance.rollback.assert_not_called()
    mock_session_instance.close.assert_not_called()
    mock_audit_tasks_logger.error.assert_called_once_with("Failed to persist audit log to database via Celery task: Database connection error")
    mock_audit_tasks_logger.info.assert_not_called()