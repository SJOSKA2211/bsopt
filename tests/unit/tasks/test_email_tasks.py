import pytest
from unittest.mock import MagicMock, patch
from src.tasks.email_tasks import send_transactional_email, send_batch_marketing_emails

@pytest.fixture
def mock_email_service():
    with patch("src.tasks.email_tasks.email_service") as mock:
        yield mock

@pytest.mark.asyncio
async def test_send_transactional_email_success(mock_email_service):
    mock_email_service.send_single_email.return_value = True
    
    with patch("src.utils.cache.rate_limiter.check_rate_limit", return_value=True):
        mock_self = MagicMock()
        # Signature: (self, to_email, subject, template_name, context)
        # It seems _orig_run might be bound or not depending on how it's accessed.
        # Let's try calling it WITHOUT self as positional.
        result = send_transactional_email._orig_run(
            "test@example.com",
            "Subject",
            "welcome.html",
            {"name": "User"}
        )
        
        assert result["status"] == "sent"
        assert mock_email_service.send_single_email.called

def test_send_batch_marketing_emails(mock_email_service):
    recipients = [{"email": "u1@ex.com"}, {"email": "u2@ex.com"}]
    send_batch_marketing_emails(recipients, "Promo", "promo.html")
    assert mock_email_service.send_batch_emails.called
