import pytest
import subprocess
from unittest.mock import patch, MagicMock
from src.database.verify import verify_postgres_connection
import os
import importlib
import sys

@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {
        "DB_PASSWORD": "testpassword",
        "DATABASE_URL": "postgresql://testuser:testpassword@localhost:5432/testdb"
    }):
        yield

def test_verify_postgres_connection_success(mock_env):
    with patch('src.database.verify.load_dotenv'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="List of databases", returncode=0)
            with patch('builtins.print'):
                verify_postgres_connection()
                mock_run.assert_called_once()
                assert os.environ["PGPASSWORD"] == "testpassword"

def test_verify_postgres_connection_no_password():
    with patch('src.database.verify.load_dotenv'):
        with patch.dict(os.environ, {}, clear=True):
            with patch('builtins.print') as mock_print:
                verify_postgres_connection()
                mock_print.assert_any_call("Error: DB_PASSWORD not found in .env")

def test_verify_postgres_connection_no_url():
    with patch('src.database.verify.load_dotenv'):
        with patch.dict(os.environ, {"DB_PASSWORD": "testpassword"}, clear=True):
            with patch('builtins.print') as mock_print:
                verify_postgres_connection()
                mock_print.assert_any_call("Error: DATABASE_URL not found in .env")

def test_verify_postgres_connection_failure(mock_env):
    with patch('src.database.verify.load_dotenv'):
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, cmd="psql", stderr="Connection refused")):
            with patch('builtins.print') as mock_print:
                verify_postgres_connection()
                mock_print.assert_any_call("\n--- Connection Failed ---")
                mock_print.assert_any_call("\nNote: Ensure the PostgreSQL container is running.")

def test_verify_postgres_connection_password_in_url():
    with patch('src.database.verify.load_dotenv'):
        with patch.dict(os.environ, {
            "DB_PASSWORD": "ignored",
            "DATABASE_URL": "postgresql://testuser:urlpass@localhost:5432/testdb"
        }):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(stdout="List of databases", returncode=0)
                with patch('builtins.print'):
                    verify_postgres_connection()
                    assert os.environ["PGPASSWORD"] == "urlpass"

def test_verify_postgres_connection_fallback_to_db_pass():
    with patch('src.database.verify.load_dotenv'):
        with patch.dict(os.environ, {
            "DB_PASSWORD": "fallbackpass",
            "DATABASE_URL": "postgresql://testuser@localhost:5432/testdb" # No pass in URL
        }):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(stdout="List of databases", returncode=0)
                with patch('builtins.print'):
                    verify_postgres_connection()
                    assert os.environ["PGPASSWORD"] == "fallbackpass"

def test_verify_main_execution():
    # Set up mocks for dependencies of verify_postgres_connection so it can run safely
    with patch('src.database.verify.load_dotenv'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="List of databases", returncode=0)
            with patch('builtins.print'):
                with patch.dict(os.environ, {
                    "DB_PASSWORD": "testpassword",
                    "DATABASE_URL": "postgresql://testuser:testpassword@localhost:5432/testdb"
                }):
                    # We use runpy to run the module as if it were the main script
                    import runpy
                    runpy.run_module('src.database.verify', run_name='__main__')
                    mock_run.assert_called_once()
