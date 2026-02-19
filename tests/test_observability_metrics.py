from unittest.mock import MagicMock, patch

from prometheus_client import REGISTRY

from src.shared.observability import push_metrics


@patch("src.shared.observability.push_to_gateway")
@patch("src.shared.observability.os.environ.get")
@patch("src.shared.observability.structlog.get_logger")
def test_push_metrics_success(mock_logger, mock_environ_get, mock_push_to_gateway):
    """Test successful pushing of metrics to Pushgateway."""
    mock_environ_get.return_value = "http://localhost:9091"  # Mock PUSHGATEWAY_URL
    mock_logger_instance = MagicMock()
    mock_logger.return_value = mock_logger_instance

    job_name = "test_job"
    result = push_metrics(job_name)

    mock_environ_get.assert_called_once_with("PUSHGATEWAY_URL")
    mock_push_to_gateway.assert_called_once_with(
        "http://localhost:9091", job=job_name, registry=REGISTRY
    )
    mock_logger_instance.info.assert_called_once_with(
        "metrics_pushed", job=job_name, gateway="http://localhost:9091"
    )
    assert result is None  # push_metrics does not return a value


@patch("src.shared.observability.push_to_gateway")
@patch("src.shared.observability.os.environ.get")
@patch("src.shared.observability.structlog.get_logger")
def test_push_metrics_no_gateway_url(
    mock_logger, mock_environ_get, mock_push_to_gateway
):
    """Test pushing metrics is skipped when PUSHGATEWAY_URL is not set."""
    mock_environ_get.return_value = None  # No PUSHGATEWAY_URL
    mock_logger_instance = MagicMock()
    mock_logger.return_value = mock_logger_instance

    job_name = "test_job"
    result = push_metrics(job_name)

    mock_environ_get.assert_called_once_with("PUSHGATEWAY_URL")
    mock_push_to_gateway.assert_not_called()
    mock_logger_instance.debug.assert_called_once_with(
        "metrics_push_skipped", reason="no_gateway_url"
    )
    assert result is None


@patch("src.shared.observability.push_to_gateway", side_effect=Exception("Push failed"))
@patch("src.shared.observability.os.environ.get")
@patch("src.shared.observability.structlog.get_logger")
def test_push_metrics_failure(mock_logger, mock_environ_get, mock_push_to_gateway):
    """Test pushing metrics when an error occurs."""
    mock_environ_get.return_value = "http://localhost:9091"
    mock_logger_instance = MagicMock()
    mock_logger.return_value = mock_logger_instance

    job_name = "test_job"
    result = push_metrics(job_name)

    mock_environ_get.assert_called_once_with("PUSHGATEWAY_URL")
    mock_push_to_gateway.assert_called_once_with(
        "http://localhost:9091", job=job_name, registry=REGISTRY
    )
    mock_logger_instance.error.assert_called_once_with(
        "metrics_push_failed", error="Push failed"
    )
    assert result is None
