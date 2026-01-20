import pytest
from datetime import datetime
from src.security.breach_notification import BreachNotificationService
from src.database.models import SecurityIncident

@pytest.mark.asyncio
async def test_report_breach_to_dpa():
    service = BreachNotificationService()
    incident = SecurityIncident(
        id="test-id",
        detected_at=datetime.utcnow(),
        nature_of_breach="Test breach",
        approximate_number_data_subjects=100,
        likely_consequences="None",
        measures_taken="None"
    )
    
    # Just verify it runs without error (simulated)
    await service.report_breach_to_dpa(incident)

@pytest.mark.asyncio
async def test_notify_affected_users():
    service = BreachNotificationService()
    incident = SecurityIncident(id="test-id")
    await service.notify_affected_users(incident, ["user1", "user2"])
