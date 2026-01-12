from unittest.mock import MagicMock, patch
import sys
import os

# Add scripts directory to path to import verify_kafka_health
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from verify_kafka_health import check_kafka_brokers, check_schema_registry, check_ksqldb

def test_check_kafka_brokers_success():
    with patch('verify_kafka_health.AdminClient') as mock_admin:
        mock_instance = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.brokers = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        mock_instance.list_topics.return_value = mock_metadata
        mock_admin.return_value = mock_instance
        
        assert check_kafka_brokers("localhost:9092") is True
        mock_instance.list_topics.assert_called_once()

def test_check_kafka_brokers_failure():
    with patch('verify_kafka_health.AdminClient') as mock_admin:
        mock_instance = MagicMock()
        mock_instance.list_topics.side_effect = Exception("Connection error")
        mock_admin.return_value = mock_instance
        
        assert check_kafka_brokers("localhost:9092") is False

def test_check_schema_registry_success():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert check_schema_registry("http://localhost:8081") is True
        mock_get.assert_called_once_with("http://localhost:8081/subjects", timeout=5)

def test_check_schema_registry_failure():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        assert check_schema_registry("http://localhost:8081") is False

def test_check_ksqldb_success_healthcheck():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert check_ksqldb("http://localhost:8088") is True
        mock_get.assert_called_with("http://localhost:8088/healthcheck", timeout=5)

def test_check_ksqldb_success_info_fallback():
    with patch('requests.get') as mock_get:
        # First call fails, second succeeds
        res1 = MagicMock()
        res1.status_code = 404
        res2 = MagicMock()
        res2.status_code = 200
        mock_get.side_effect = [res1, res2]
        
        assert check_ksqldb("http://localhost:8088") is True
        assert mock_get.call_count == 2

def test_check_ksqldb_failure():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        assert check_ksqldb("http://localhost:8088") is False
