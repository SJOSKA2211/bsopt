from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.database.models import SecurityIncident
from src.security.breach_notification import BreachNotificationService


@pytest.mark.asyncio
@patch("src.services.email_service.SendGridAPIClient")
async def test_report_breach_to_dpa(mock_sendgrid):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_sendgrid.return_value.send.return_value = mock_response

    service = BreachNotificationService()
    incident = SecurityIncident(
        id="test-id",
        detected_at=datetime.now(UTC),
        nature_of_breach="Test breach",
        approximate_number_data_subjects=100,
        likely_consequences="None",
        measures_taken="None",
        data_categories_affected=["email", "password"]
    )
    
    success = await service.report_breach_to_dpa(incident)
    
    assert success is True
    assert mock_sendgrid.return_value.send.called

@pytest.mark.asyncio
@patch("src.services.email_service.SendGridAPIClient")
async def test_notify_affected_users(mock_sendgrid):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_sendgrid.return_value.send.return_value = mock_response

    service = BreachNotificationService()
    incident = SecurityIncident(
        id="test-id",
        event_type="Unauthorized Access",
        measures_taken="Password Reset"
    )
    
    user1 = MagicMock(email="user1@example.com", full_name="User One")
    user2 = MagicMock(email="user2@example.com", full_name="User Two")
    
    count = await service.notify_affected_users(incident, [user1, user2])
    
    assert count == 2
    assert mock_sendgrid.return_value.send.called
