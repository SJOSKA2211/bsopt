import pytest
from unittest.mock import MagicMock, patch, ANY
from sqlalchemy.engine import Engine
from sqlalchemy import text # Import text
import src.database # Import the module to ensure _engine and _SessionLocal are initialized as None

from src.database import (
    get_db, 
    get_db_context, 
    get_session, 
    get_engine, 
    get_pool_status, 
    health_check, 
    dispose_engine,
    create_tables,
    drop_tables
)
from src.config import settings

@pytest.fixture(autouse=True)
def mock_db_components_for_tests():
    mock_engine_instance = MagicMock(spec=Engine)
    mock_session_local_instance = MagicMock()
    mock_session_instance = MagicMock() # The actual session mock that will be returned

    # Set the return value of mock_session_local_instance to mock_session_instance
    # This means when _SessionLocal() is called, it returns mock_session_instance
    mock_session_local_instance.return_value = mock_session_instance

    # Mock the 'pool' attribute and its methods for the engine mock
    mock_pool = MagicMock()
    mock_pool.size.return_value = 5
    mock_pool.checkedout.return_value = 2
    mock_pool.overflow.return_value = 0
    mock_pool.checkedin.return_value = 5
    mock_engine_instance.pool = mock_pool

    with patch("src.database.create_engine", return_value=mock_engine_instance) as mock_create_engine, \
         patch("src.database.sessionmaker", return_value=mock_session_local_instance) as mock_sessionmaker, \
         patch("src.database._initialize_db_components") as mock_initialize, \
         patch.object(src.database, "_engine", new=mock_engine_instance), \
         patch.object(src.database, "_SessionLocal", new=mock_session_local_instance), \
         patch.object(src.database, "_SLOW_QUERY_THRESHOLD_MS", new=100), \
         patch("src.database.event.listen") as mock_event_listen, \
         patch("src.config.settings.DATABASE_URL", "sqlite:///test.db"):

        # The _initialize_db_components function should do nothing when mocked
        mock_initialize.return_value = None

        yield mock_engine_instance, mock_session_instance

    # Clear _engine and _SessionLocal after tests to ensure clean state for other test files
    src.database._engine = None
    src.database._SessionLocal = None
    src.database._SLOW_QUERY_THRESHOLD_MS = None

def test_get_engine(mock_db_components_for_tests):
    mock_engine_instance, _ = mock_db_components_for_tests
    engine = get_engine()
    assert engine is mock_engine_instance

def test_get_session(mock_db_components_for_tests):
    _, mock_session_instance = mock_db_components_for_tests
    session = get_session()
    assert session is mock_session_instance

def test_get_db(mock_db_components_for_tests):
    _, mock_session_instance = mock_db_components_for_tests
    gen = get_db()
    session = next(gen)
    assert session is mock_session_instance
    with pytest.raises(StopIteration):
        next(gen)
    mock_session_instance.close.assert_called_once()


def test_get_db_context(mock_db_components_for_tests):
    _, mock_session_instance = mock_db_components_for_tests
    with get_db_context() as session:
        assert session is mock_session_instance
    mock_session_instance.commit.assert_called_once()
    mock_session_instance.close.assert_called_once()


def test_get_pool_status(mock_db_components_for_tests):
    mock_engine_instance, _ = mock_db_components_for_tests
    # The pool attribute is already mocked in the fixture setup
    status = get_pool_status()
    assert "pool_size" in status
    assert status["pool_size"] == 5
    assert status["checked_out"] == 2


def test_health_check(mock_db_components_for_tests):
    mock_engine_instance, _ = mock_db_components_for_tests
    mock_connect = MagicMock()
    mock_engine_instance.connect.return_value.__enter__.return_value = mock_connect
    assert health_check() is True
    # Use ANY to match the TextClause object
    mock_connect.execute.assert_called_once_with(ANY) 
    # Optionally, if more strict: assert str(mock_connect.execute.call_args.args[0]) == "SELECT 1"

def test_dispose_engine(mock_db_components_for_tests):
    mock_engine_instance, _ = mock_db_components_for_tests
    dispose_engine()
    mock_engine_instance.dispose.assert_called_once()

@patch("src.database.settings")
@patch("src.database.Base.metadata.create_all")
def test_create_tables_blocked(mock_create_all, mock_settings, mock_db_components_for_tests):
    mock_settings.ENVIRONMENT = "production"
    create_tables()
    mock_create_all.assert_not_called()

@patch("src.database.settings")
@patch("src.database.Base.metadata.drop_all")
def test_drop_tables_blocked(mock_drop_all, mock_settings, mock_db_components_for_tests):
    mock_settings.ENVIRONMENT = "production"
    drop_tables()
    mock_drop_all.assert_not_called()

@patch("src.database.settings")
@patch("src.database.Base.metadata.create_all")
def test_create_tables_dev(mock_create_all, mock_settings, mock_db_components_for_tests):
    mock_settings.ENVIRONMENT = "test"
    create_tables()
    mock_create_all.assert_called_once()

@patch("src.database.settings")
@patch("src.database.Base.metadata.drop_all")
def test_drop_tables_dev(mock_drop_all, mock_settings, mock_db_components_for_tests):
    mock_settings.ENVIRONMENT = "test"
    drop_tables()
    mock_drop_all.assert_called_once()
