import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.security.breach_notification import BreachNotificationService
from src.database.models import SecurityIncident

@pytest.fixture(autouse=True)
def mock_sendgrid():
    with patch("src.services.email_service.SendGridAPIClient") as mock:
        mock_client = mock.return_value
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_client.send.return_value = mock_response
        yield mock

@pytest.fixture(autouse=True)
def mock_jinja(autouse=True):
    with patch("src.services.email_service.Environment") as mock_env:
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Email Content</html>"
        mock_env.return_value.get_template.return_value = mock_template
        yield mock_env

@pytest.mark.asyncio
async def test_report_breach_to_dpa():
    # Use patch to ensure settings don't trigger errors if missing
    with patch("src.security.breach_notification.settings") as mock_settings:
        mock_settings.SENDGRID_API_KEY = "test_key"
        mock_settings.DEFAULT_FROM_EMAIL = "from@example.com"
        mock_settings.DPA_EMAIL = "dpa@example.com"
        
        service = BreachNotificationService()
        incident = SecurityIncident(
            id="test-id",
            detected_at=datetime.now(),
            nature_of_breach="Test breach",
            approximate_number_data_subjects=100,
            approximate_number_records=500,
            data_categories_affected=["email", "password"],
            likely_consequences="None",
            measures_taken="None"
        )
        
        # Just verify it runs without error (simulated)
        result = await service.report_breach_to_dpa(incident)
        assert result is True

@pytest.mark.asyncio
async def test_notify_affected_users():
    with patch("src.security.breach_notification.settings") as mock_settings:
        mock_settings.SENDGRID_API_KEY = "test_key"
        mock_settings.DEFAULT_FROM_EMAIL = "from@example.com"
        
        service = BreachNotificationService()
        incident = SecurityIncident(id="test-id", incident_type="Data Breach", measures_taken="None")
        
        # Mock User objects
        user1 = MagicMock()
        user1.email = "user1@example.com"
        user1.full_name = "User One"
        
        user2 = MagicMock()
        user2.email = "user2@example.com"
        user2.full_name = "User Two"
        
        count = await service.notify_affected_users(incident, [user1, user2])
        assert count == 2