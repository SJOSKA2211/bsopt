import pytest
from unittest.mock import MagicMock, patch


from src.aiops.prometheus_adapter import PrometheusClient

TEST_PROMETHEUS_URL = "http://test-prometheus:9090"
TEST_SERVICE = "test-service"

def test_prometheus_client_class_exists():
    """
    Test that the PrometheusClient class can be imported.
    """
    assert PrometheusClient is not None, "PrometheusClient class is not defined or importable."

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
def test_prometheus_client_init(mock_prometheus_connect):
    """
    Test that PrometheusClient constructor initializes PrometheusConnect correctly.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance

    client = PrometheusClient(TEST_PROMETHEUS_URL)

    mock_prometheus_connect.assert_called_once_with(url=TEST_PROMETHEUS_URL, disable_ssl=True)
    assert client.prom == mock_connect_instance

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_check_connectivity_success(mock_logger, mock_prometheus_connect):
    """
    Test check_connectivity logs success on Prometheus being reachable.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.all_metrics.return_value = ["metric1"]

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    client.check_connectivity()

    mock_connect_instance.all_metrics.assert_called_once()
    mock_logger.info.assert_called_once_with("prometheus_connectivity_ok", url=TEST_PROMETHEUS_URL)
    mock_logger.error.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_check_connectivity_failure(mock_logger, mock_prometheus_connect):
    """
    Test check_connectivity logs failure and raises exception on Prometheus being unreachable.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.all_metrics.side_effect = Exception("Prometheus down")

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    with pytest.raises(Exception, match="Prometheus down"):
        client.check_connectivity()

    mock_connect_instance.all_metrics.assert_called_once()
    mock_logger.error.assert_called_once_with("prometheus_connectivity_failed", url=TEST_PROMETHEUS_URL, error="Prometheus down")
    mock_logger.info.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_5xx_error_rate_success(mock_logger, mock_prometheus_connect):
    """
    Test get_5xx_error_rate returns correct value on success.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.return_value = [{'value': [0, '0.05']}]

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    error_rate = client.get_5xx_error_rate(TEST_SERVICE)

    expected_query = f'sum(rate(http_requests_total{{status=~"5..", service="{TEST_SERVICE}"}}[5m])) / sum(rate(http_requests_total{{service="{TEST_SERVICE}"}}[5m]))'
    mock_connect_instance.custom_query.assert_called_once_with(query=expected_query)
    assert error_rate == 0.05
    mock_logger.error.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_5xx_error_rate_empty_service(mock_logger, mock_prometheus_connect):
    """
    Test get_5xx_error_rate raises ValueError for empty service name.
    """
    client = PrometheusClient(TEST_PROMETHEUS_URL)
    with pytest.raises(ValueError, match="Service name cannot be empty"):
        client.get_5xx_error_rate("")
    mock_logger.error.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_5xx_error_rate_query_failure(mock_logger, mock_prometheus_connect):
    """
    Test get_5xx_error_rate logs error and returns 0.0 on query failure.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.side_effect = Exception("Query error")

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    error_rate = client.get_5xx_error_rate(TEST_SERVICE)

    expected_query = f'sum(rate(http_requests_total{{status=~"5..", service="{TEST_SERVICE}"}}[5m])) / sum(rate(http_requests_total{{service="{TEST_SERVICE}"}}[5m]))'
    mock_connect_instance.custom_query.assert_called_once_with(query=expected_query)
    mock_logger.error.assert_called_once_with("fetch_5xx_failed", service=TEST_SERVICE, error="Query error", query=expected_query)
    assert error_rate == 0.0

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_5xx_error_rate_empty_result(mock_logger, mock_prometheus_connect):
    """
    Test get_5xx_error_rate returns 0.0 on empty result.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.return_value = []

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    error_rate = client.get_5xx_error_rate(TEST_SERVICE)
    assert error_rate == 0.0

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_p95_latency_success(mock_logger, mock_prometheus_connect):
    """
    Test get_p95_latency returns correct value on success.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.return_value = [{'value': [0, '0.123']}]

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    latency = client.get_p95_latency(TEST_SERVICE)

    expected_query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{TEST_SERVICE}"}}[5m])) by (le))'
    mock_connect_instance.custom_query.assert_called_once_with(query=expected_query)
    assert latency == 0.123
    mock_logger.error.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_p95_latency_empty_service(mock_logger, mock_prometheus_connect):
    """
    Test get_p95_latency raises ValueError for empty service name.
    """
    client = PrometheusClient(TEST_PROMETHEUS_URL)
    with pytest.raises(ValueError, match="Service name cannot be empty"):
        client.get_p95_latency("")
    mock_logger.error.assert_not_called()

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_p95_latency_query_failure(mock_logger, mock_prometheus_connect):
    """
    Test get_p95_latency logs error and returns 0.0 on query failure.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.side_effect = Exception("Query error")

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    latency = client.get_p95_latency(TEST_SERVICE)

    expected_query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{TEST_SERVICE}"}}[5m])) by (le))'
    mock_connect_instance.custom_query.assert_called_once_with(query=expected_query)
    mock_logger.error.assert_called_once_with("fetch_p95_failed", service=TEST_SERVICE, error="Query error", query=expected_query)
    assert latency == 0.0

@patch('src.aiops.prometheus_adapter.PrometheusConnect')
@patch('src.aiops.prometheus_adapter.logger')
def test_get_p95_latency_empty_result(mock_logger, mock_prometheus_connect):
    """
    Test get_p95_latency returns 0.0 on empty result.
    """
    mock_connect_instance = MagicMock()
    mock_prometheus_connect.return_value = mock_connect_instance
    mock_connect_instance.custom_query.return_value = []

    client = PrometheusClient(TEST_PROMETHEUS_URL)
    latency = client.get_p95_latency(TEST_SERVICE)
    assert latency == 0.0
