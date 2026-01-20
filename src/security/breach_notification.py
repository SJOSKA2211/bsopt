"""
Breach Notification Service
===========================

Service for handling data breach notifications as required by GDPR.
"""

import logging
from datetime import datetime, timezone
from typing import List
from uuid import UUID

from src.config import settings
from src.database.models import SecurityIncident, User
from src.services.email_service import TransactionalEmailService

logger = logging.getLogger(__name__)


class BreachNotificationService:
    """
    Service for handling data breach notifications.
    """

    def __init__(self):
        self.email_service = TransactionalEmailService(
            api_key=settings.SENDGRID_API_KEY,
            from_email=settings.DEFAULT_FROM_EMAIL
        )
        self.DPA_EMAIL = settings.DPA_EMAIL
        self.DPA_CONTACT_PERSON = "Data Protection Officer"

    async def report_breach_to_dpa(self, incident: SecurityIncident) -> bool:
        """
        Report to Data Protection Authority within 72 hours.
        """
        # 1. GDPR Compliance Check
        time_elapsed = (datetime.now(timezone.utc) - incident.detected_at.astimezone(timezone.utc)).total_seconds()
        if time_elapsed > 72 * 3600:
            logger.critical(f"GDPR VIOLATION: 72-hour notification deadline missed for incident {incident.id}")

        # 2. Prepare Context for DPA
        context = {
            "incident_id": str(incident.id),
            "nature": incident.nature_of_breach,
            "data_subjects": incident.approximate_number_data_subjects,
            "records_count": incident.approximate_number_records,
            "categories": ", ".join(incident.data_categories_affected),
            "consequences": incident.likely_consequences,
            "measures": incident.measures_taken,
            "detected_at": incident.detected_at.isoformat(),
            "contact_person": self.DPA_CONTACT_PERSON
        }

        # 3. Send Notification
        success = self.email_service.send_single_email(
            to_email=self.DPA_EMAIL,
            subject=f"[URGENT] GDPR Data Breach Notification - Ref: {incident.id}",
            template_name="dpa_notification.html",
            context=context,
            email_type="compliance"
        )

        if success:
            logger.info(f"Successfully reported breach {incident.id} to DPA at {self.DPA_EMAIL}")
        else:
            logger.error(f"Failed to report breach {incident.id} to DPA")
        
        return success

    async def notify_affected_users(self, incident: SecurityIncident, users: List[User]) -> int:
        """
        Notify data subjects if there is a high risk to their rights and freedoms.
        GDPR Article 34 requirement.
        """
        if not users:
            return 0

        recipients = []
        for user in users:
            recipients.append({
                "email": user.email,
                "context": {
                    "user_name": user.full_name or "Valued User",
                    "incident_type": incident.incident_type,
                    "measures_taken": incident.measures_taken,
                    "recommendation": "We recommend changing your password and enabling MFA immediately."
                }
            })

        # Send batch emails (max 1000 per batch handled by SendGrid)
        # We chunk if necessary, but send_batch_emails handles the base batch
        success = self.email_service.send_batch_emails(
            recipients=recipients,
            subject="Security Alert: Important information regarding your BS-Opt account",
            template_name="user_breach_alert.html",
            email_type="security"
        )

        if success:
            logger.info(f"Sent security notifications to {len(users)} users for incident {incident.id}")
            return len(users)
        else:
            logger.error(f"Failed to send batch security notifications for incident {incident.id}")
            return 0


breach_notification_service = BreachNotificationService()
