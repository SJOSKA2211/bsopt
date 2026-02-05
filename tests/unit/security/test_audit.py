from unittest.mock import MagicMock, patch

from src.database.models import User
from src.security.audit import AuditEvent, log_audit


@patch("src.tasks.audit_tasks.persist_audit_log.delay")
def test_log_audit_basic(mock_delay):
    user = User(id="test-uid", email="test@example.com")
    log_audit(AuditEvent.USER_LOGIN_SUCCESS, user=user, persist_to_db=True)
    
    assert mock_delay.called
    args, kwargs = mock_delay.call_args
    assert kwargs["event_type"] == AuditEvent.USER_LOGIN_SUCCESS.value
    assert kwargs["user_id"] == "test-uid"

def test_log_audit_with_request():
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.headers = {"user-agent": "test-agent"}
    mock_request.url.path = "/api/v1/auth/login"
    mock_request.method = "POST"
    
    with patch("src.security.audit.audit_logger") as mock_logger:
        log_audit(AuditEvent.SUSPICIOUS_ACTIVITY, request=mock_request, persist_to_db=False)
        assert mock_logger.info.called
        log_data = mock_logger.info.call_args[0][0]
        assert log_data["source_ip"] == "127.0.0.1"
        assert log_data["event_type"] == "SUSPICIOUS_ACTIVITY"
