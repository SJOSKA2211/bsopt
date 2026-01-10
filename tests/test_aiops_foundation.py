import pytest
import os
from src.aiops.prometheus_client import PrometheusClient

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
