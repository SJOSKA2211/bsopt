import requests

def test_get_metrics_returns_prometheus_metrics_text():
    url = "http://127.0.0.1:8000/metrics"
    headers = {
        "Accept": "text/plain"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /metrics failed with exception: {e}"

    # Assert status code is 200
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    content_type = response.headers.get("Content-Type", "")
    # Acceptable content types for Prometheus metrics text format
    acceptable_cts = [
        "text/plain; version=0.0.4; charset=utf-8",
        "text/plain; charset=utf-8",
        "text/plain"
    ]
    assert any(content_type.startswith(ct) for ct in acceptable_cts), f"Unexpected Content-Type header: {content_type}"

    # Check that response text contains some Prometheus metric-like content (e.g., HELP or TYPE lines)
    text = response.text
    assert text.startswith("# HELP") or "# HELP" in text, "Prometheus metrics text missing expected '# HELP' line"
    assert "# TYPE" in text, "Prometheus metrics text missing expected '# TYPE' line"

test_get_metrics_returns_prometheus_metrics_text()