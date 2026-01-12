import pytest
from src.aiops.prometheus_adapter import PrometheusClient

def test_prometheus_client_init():
    """Test initialization of PrometheusClient."""
    url = "http://localhost:9090"
    client = PrometheusClient(url=url)
    assert client.url == url

def test_prometheus_client_connectivity_fail():
    """Test that connectivity fails if Prometheus is unreachable (Red Phase)."""
    # Using a definitely non-existent local port
    url = "http://localhost:9999"
    client = PrometheusClient(url=url)
    with pytest.raises(Exception):
        client.check_connectivity()

def test_prometheus_client_connectivity_success(mocker):
    """Test that connectivity succeeds if Prometheus is reachable."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    client = PrometheusClient(url="http://localhost:9090")
    client.check_connectivity()
    mock_prom.return_value.all_metrics.assert_called_once()

def test_fetch_5xx_errors(mocker):
    """Test fetching 5xx error rate from Prometheus."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    # Mock return value for custom_query
    mock_prom.return_value.custom_query.return_value = [
        {"value": [1234567890, "0.05"]}
    ]
    
    client = PrometheusClient(url="http://localhost:9090")
    rate = client.get_5xx_error_rate(service="api")
    
    assert rate == 0.05
    mock_prom.return_value.custom_query.assert_called_once()

def test_fetch_5xx_errors_fail(mocker):
    """Test fetching 5xx error rate failure."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    mock_prom.return_value.custom_query.side_effect = Exception("Query failed")
    
    client = PrometheusClient(url="http://localhost:9090")
    rate = client.get_5xx_error_rate(service="api")
    
    assert rate == 0.0
    mock_prom.return_value.custom_query.assert_called_once()

def test_fetch_5xx_errors_empty(mocker):
    """Test fetching 5xx error rate with empty result."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    mock_prom.return_value.custom_query.return_value = []
    
    client = PrometheusClient(url="http://localhost:9090")
    rate = client.get_5xx_error_rate(service="api")
    
    assert rate == 0.0

def test_fetch_p95_latency(mocker):
    """Test fetching p95 latency from Prometheus."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    mock_prom.return_value.custom_query.return_value = [
        {"value": [1234567890, "0.15"]}
    ]
    
    client = PrometheusClient(url="http://localhost:9090")
    latency = client.get_p95_latency(service="api")
    
    assert latency == 0.15
    mock_prom.return_value.custom_query.assert_called_once()

def test_fetch_p95_latency_fail(mocker):
    """Test fetching p95 latency failure."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    mock_prom.return_value.custom_query.side_effect = Exception("Query failed")
    
    client = PrometheusClient(url="http://localhost:9090")
    latency = client.get_p95_latency(service="api")
    
    assert latency == 0.0

def test_fetch_p95_latency_empty(mocker):
    """Test fetching p95 latency with empty result."""
    mock_prom = mocker.patch("src.aiops.prometheus_adapter.PrometheusConnect")
    mock_prom.return_value.custom_query.return_value = []
    
    client = PrometheusClient(url="http://localhost:9090")
    latency = client.get_p95_latency(service="api")
    
    assert latency == 0.0

def test_fetch_5xx_errors_invalid_service():
    """Test fetching 5xx error rate with invalid service name."""
    client = PrometheusClient(url="http://localhost:9090")
    with pytest.raises(ValueError, match="Service name cannot be empty"):
        client.get_5xx_error_rate(service="")

def test_fetch_p95_latency_invalid_service():
    """Test fetching p95 latency with invalid service name."""
    client = PrometheusClient(url="http://localhost:9090")
    with pytest.raises(ValueError, match="Service name cannot be empty"):
        client.get_p95_latency(service="")
