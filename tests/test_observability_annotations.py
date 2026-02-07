from datetime import UTC
from unittest.mock import ANY, MagicMock, patch

import httpx

from src.shared.observability import post_grafana_annotation  # Assuming new function


@patch("src.shared.observability.httpx.post")
@patch("src.shared.observability.os.environ.get")
def test_post_grafana_annotation_success(mock_environ_get, mock_httpx_post):
    """Test successful posting of a Grafana annotation."""
    mock_environ_get.return_value = "http://localhost:3000"  # Mock GRAFANA_URL
    mock_httpx_post.return_value.status_code = 200
    mock_httpx_post.return_value.json.return_value = {"message": "Annotation added"}

    message = "Service restarted due to high error rate."
    tags = ["remediation", "service_restart"]

    result = post_grafana_annotation(message, tags)
    assert result
    mock_environ_get.assert_called_once_with("GRAFANA_URL")
    mock_httpx_post.assert_called_once_with(
        "http://localhost:3000/api/annotations",
        headers={"Content-Type": "application/json"},
        json={"time": ANY, "text": message, "tags": tags},  # Use ANY for time
        timeout=5,
    )


@patch("src.shared.observability.httpx.post")
@patch("src.shared.observability.os.environ.get")
def test_post_grafana_annotation_no_grafana_url(mock_environ_get, mock_httpx_post):
    """Test posting annotation when GRAFANA_URL is not set."""
    mock_environ_get.return_value = None  # No GRAFANA_URL

    message = "Test message"
    tags = ["test"]

    result = post_grafana_annotation(message, tags)
    assert not result
    mock_environ_get.assert_called_once_with("GRAFANA_URL")
    mock_httpx_post.assert_not_called()


@patch("src.shared.observability.httpx.post")
@patch("src.shared.observability.os.environ.get")
def test_post_grafana_annotation_api_failure(mock_environ_get, mock_httpx_post):
    """Test posting annotation when Grafana API returns an error."""
    mock_environ_get.return_value = "http://localhost:3000"
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=MagicMock(), response=mock_response
    )
    mock_httpx_post.return_value = mock_response

    message = "Test message"
    tags = ["test"]

    result = post_grafana_annotation(message, tags)
    assert not result
    mock_environ_get.assert_called_once_with("GRAFANA_URL")
    mock_httpx_post.assert_called_once()


@patch("src.shared.observability.httpx.post", side_effect=Exception("Connection error"))
@patch("src.shared.observability.os.environ.get")
def test_post_grafana_annotation_connection_error(mock_environ_get, mock_httpx_post):
    """Test posting annotation when a connection error occurs."""
    mock_environ_get.return_value = "http://localhost:3000"

    message = "Test message"
    tags = ["test"]

    result = post_grafana_annotation(message, tags)
    assert not result
    mock_environ_get.assert_called_once_with("GRAFANA_URL")
    mock_httpx_post.assert_called_once()


@patch("src.shared.observability.datetime")
@patch("src.shared.observability.httpx.post")
@patch("src.shared.observability.os.environ.get")
def test_post_grafana_annotation_payload_time(
    mock_environ_get, mock_httpx_post, mock_datetime
):
    """Test that the time in the payload is correctly formatted."""
    mock_environ_get.return_value = "http://localhost:3000"
    mock_httpx_post.return_value.status_code = 200
    mock_httpx_post.return_value.json.return_value = {"message": "Annotation added"}

    # Mock datetime.now(timezone.utc) and timestamp()
    mock_now = MagicMock()
    mock_now.timestamp.return_value = 1672531200.0  # Example Unix timestamp
    mock_datetime.now.return_value = mock_now

    message = "Test message"
    tags = ["test"]

    post_grafana_annotation(message, tags)

    expected_payload_time = int(mock_now.timestamp.return_value * 1000)

    mock_httpx_post.assert_called_once()
    args, kwargs = mock_httpx_post.call_args
    assert kwargs["json"]["time"] == expected_payload_time
    mock_datetime.now.assert_called_once_with(UTC)
