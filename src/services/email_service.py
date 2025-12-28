import logging
import os
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape
from prometheus_client import Counter, Histogram
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, Mail, Personalization

logger = logging.getLogger(__name__)

# Metrics
EMAILS_SENT_TOTAL = Counter("emails_sent_total", "Total emails sent", ["status", "type"])
EMAIL_DELIVERY_TIME = Histogram("email_delivery_time_seconds", "Time taken to send email", ["type"])


class TransactionalEmailService:
    """
    Service for sending transactional emails via SendGrid.
    Supports Jinja2 templating, batching, and monitoring.
    """

    def __init__(self, api_key: str, from_email: str):
        self.client = SendGridAPIClient(api_key)
        self.from_email = from_email

        # Setup Jinja2 template engine
        template_dir = os.path.join(os.getcwd(), "src", "templates", "emails")
        os.makedirs(template_dir, exist_ok=True)
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir), autoescape=select_autoescape(["html", "xml"])
        )

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render email template with context."""
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def send_single_email(
        self,
        to_email: str,
        subject: str,
        template_name: str,
        context: Dict[str, Any],
        email_type: str = "transactional",
    ) -> bool:
        """Send a single email with monitoring and error handling."""
        html_content = self._render_template(template_name, context)

        message = Mail(
            from_email=self.from_email,
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )

        with EMAIL_DELIVERY_TIME.labels(type=email_type).time():
            try:
                response = self.client.send(message)
                if response.status_code >= 200 and response.status_code < 300:
                    EMAILS_SENT_TOTAL.labels(status="success", type=email_type).inc()
                    return True
                else:
                    logger.error(f"SendGrid error: {response.status_code} - {response.body}")
                    EMAILS_SENT_TOTAL.labels(status="error", type=email_type).inc()
                    return False
            except Exception as e:
                logger.error(f"Failed to send email to {to_email}: {e}")
                EMAILS_SENT_TOTAL.labels(status="failed", type=email_type).inc()
                raise  # Re-raise for Celery retry

    def send_batch_emails(
        self,
        recipients: List[Dict[str, Any]],
        subject: str,
        template_name: str,
        email_type: str = "marketing",
    ) -> bool:
        """
        Send batch emails using SendGrid Personalizations.
        Max 1000 recipients per batch.
        """
        if not recipients:
            return True

        message = Mail(from_email=self.from_email, subject=subject)

        for recipient in recipients:
            p = Personalization()
            p.add_to(Email(recipient["email"]))

            # Render specific content if needed or use global template
            # For simplicity, we use one template for the whole batch
            html_content = self._render_template(template_name, recipient.get("context", {}))
            message.content = (
                html_content  # Note: In batch, usually use dynamic templates on SendGrid side
            )

            message.add_personalization(p)

        try:
            response = self.client.send(message)
            return bool(response.status_code < 300)
        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            return False
