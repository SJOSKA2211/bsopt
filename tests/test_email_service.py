from unittest.mock import MagicMock, patch

from src.services.email_service import TransactionalEmailService


def test_email_service_send():
    with patch("src.services.email_service.SendGridAPIClient") as mock_sg:
        mock_sg.return_value.send.return_value.status_code = 202

        service = TransactionalEmailService(api_key="fake", from_email="from@test.com")
        # Mock template rendering
        service._render_template = MagicMock(return_value="<html></html>")

        success = service.send_single_email(
            to_email="to@test.com",
            subject="Test",
            template_name="test.html",
            context={},
        )
        assert success
