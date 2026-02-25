from unittest.mock import MagicMock, patch

import pytest

from src.tasks.email_tasks import send_batch_marketing_emails, send_transactional_email


@pytest.fixture
def mock_rate_limiter():
    with patch("src.tasks.email_tasks.rate_limiter") as mock:
        yield mock


@pytest.fixture
def mock_email_service():
    with patch("src.tasks.email_tasks.email_service") as mock:
        yield mock


def test_send_transactional_email_success(mock_rate_limiter, mock_email_service):
    # Call the original function to bypass Celery decorator mess
    # Celery tasks usually store the original function in .__wrapped__ or ._orig_run
    orig_func = getattr(send_transactional_email, "_orig_run", send_transactional_email)
    if hasattr(orig_func, "__wrapped__"):
        orig_func = orig_func.__wrapped__

    with patch("asyncio.run", return_value=True):
        mock_email_service.send_single_email.return_value = True

        # Celery bound tasks expect 'self' as first arg
        res = orig_func(
            MagicMock(),  # self
            to_email="morty@jerry.com",
            subject="Test",
            template_name="test",
            context={},
        )

        assert res["status"] == "sent"


def test_send_batch_marketing_emails_success(mock_email_service):
    res = send_batch_marketing_emails(recipients=["a@b.com"], subject="S", template_name="t")
    assert res["status"] == "batch_sent"
