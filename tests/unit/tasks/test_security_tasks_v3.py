from unittest.mock import MagicMock, patch

import pytest

from src.workers.tasks.security_tasks import rehash_legacy_passwords


@pytest.fixture
def mock_db():
    with patch("src.workers.tasks.security_tasks.get_db_session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


def test_rehash_legacy_passwords_success(mock_db):
    mock_user = MagicMock()
    mock_user.hashed_password = "bcrypt_hash"
    mock_db.execute.return_value.scalars.return_value.all.return_value = [mock_user]

    with patch("src.workers.tasks.security_tasks.get_password_service") as mock_pw_service:
        mock_pw_service.return_value.needs_rehash.return_value = True

        # Use .run or .__wrapped__
        orig_func = getattr(rehash_legacy_passwords, "_orig_run", rehash_legacy_passwords)

        res = orig_func(MagicMock())  # self
        assert res["status"] == "completed"
        assert mock_db.commit.called


def test_rehash_legacy_passwords_failure(mock_db):
    mock_db.execute.side_effect = Exception("Select fail")

    task_mock = MagicMock()
    orig_func = getattr(rehash_legacy_passwords, "_orig_run", rehash_legacy_passwords)

    with pytest.raises(Exception):
        orig_func(task_mock)  # self

    assert mock_db.rollback.called
    assert task_mock.retry.called