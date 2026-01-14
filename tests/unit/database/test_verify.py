import pytest
from unittest.mock import patch, MagicMock
from src.database.verify import verify_postgres_connection

@patch("subprocess.run")
@patch("os.getenv")
def test_verify_postgres_connection_success(mock_getenv, mock_run):
    mock_getenv.return_value = "password"
    mock_run.return_value = MagicMock(stdout="database list")
    
    verify_postgres_connection()
    assert mock_run.called

@patch("subprocess.run")
@patch("os.getenv")
def test_verify_postgres_connection_failure(mock_getenv, mock_run):
    from subprocess import CalledProcessError
    mock_getenv.return_value = "password"
    mock_run.side_effect = CalledProcessError(1, "psql", stderr="Connection refused")
    
    verify_postgres_connection()
    assert mock_run.called
