import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.shared.db import Base, MarketData

@pytest.fixture
def mock_engine():
    return MagicMock()

@pytest.fixture
def mock_session():
    return MagicMock()

def test_market_data_model():
    """Verify that the MarketData model has the correct columns."""
    assert MarketData.__tablename__ == 'market_data'
    assert hasattr(MarketData, 'id')
    assert hasattr(MarketData, 'ticker')
    assert hasattr(MarketData, 'timestamp')
    assert hasattr(MarketData, 'open')
    assert hasattr(MarketData, 'high')
    assert hasattr(MarketData, 'low')
    assert hasattr(MarketData, 'close')
    assert hasattr(MarketData, 'volume')

@patch('src.shared.db.create_engine')
@patch('src.shared.db.sessionmaker')
def test_db_connection(mock_sessionmaker, mock_create_engine):
    """Verify that the database connection is established correctly."""
    from src.shared.db import get_db_session
    
    mock_create_engine.return_value = MagicMock()
    mock_sessionmaker.return_value = MagicMock()
    
    session = get_db_session("postgresql://user:pass@localhost/db")
    
    mock_create_engine.assert_called_with("postgresql://user:pass@localhost/db")
    mock_sessionmaker.assert_called_once()
