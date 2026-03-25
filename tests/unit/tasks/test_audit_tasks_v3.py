from unittest.mock import MagicMock, patch

import pytest

from src.workers.tasks.audit_tasks import persist_audit_log

@pytest.fixture
def mock_session():
    with patch("src.workers.tasks.audit_tasks.get_session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session

def test_persist_audit_log_success(mock_session):
    persist_audit_log(
        event_type="LOGIN",
        user_id="engineer",
        user_email="engineer@bsopt.com",
        source_ip="127.0.0.1",
        user_agent="EnterpriseBrowser",
        request_path="/login",
        request_method="POST",
        details={"dimension": "C-137"},
    )

    assert mock_session.add.called
    assert mock_session.commit.called
    assert mock_session.close.called

def test_persist_audit_log_failure(mock_session):
    mock_session.commit.side_effect = Exception("DB Boom!")

    # Should handle exception internally
    persist_audit_log(
        event_type="BOOM",
        user_id="assistant",
        user_email=None,
        source_ip=None,
        user_agent=None,
        request_path=None,
        request_method=None,
        details=None,
    )

    assert mock_session.rollback.called
    assert mock_session.close.called
