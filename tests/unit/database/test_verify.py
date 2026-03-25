from unittest.mock import MagicMock, patch

from src.database.verify import verify_connection


@patch("src.database.verify.get_engine")
@patch("src.database.verify.get_settings")
def test_verify_connection_success(mock_get_settings, mock_get_engine):
    # Mock settings
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    
    # Mock engine and connection
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    
    # Mock query results
    mock_conn.execute.return_value.scalar.return_value = 1
    
    verify_connection()
    assert mock_conn.execute.called

@patch("src.database.verify.get_engine")
@patch("src.database.verify.get_settings")
def test_verify_connection_failure(mock_get_settings, mock_get_engine):
    mock_get_settings.side_effect = Exception("Settings load failed")
    
    try:
        verify_connection()
    except SystemExit as e:
        assert e.code == 1
