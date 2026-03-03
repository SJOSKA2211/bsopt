import requests


def test_get_metrics_returns_prometheus_metrics_text():
    url = "http://127.0.0.1:8000/metrics"
    headers = {"Accept": "text/plain"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to /metrics failed: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    content_type = response.headers.get("Content-Type", "")
    assert "text/plain" in content_type, (
        f"Expected 'text/plain' in Content-Type, got '{content_type}'"
    )
    response_text = response.text
    assert response_text is not None and len(response_text) > 0, "Response text is empty"
    # Basic check for Prometheus metrics format: lines with metric name and values, e.g. "metric_name 123"
    lines = response_text.splitlines()
    assert any(line and not line.startswith("#") and " " in line for line in lines), (
        "Response does not contain Prometheus metrics format"
    )


test_get_metrics_returns_prometheus_metrics_text()
