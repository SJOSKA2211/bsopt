import requests

def test_get_metrics_returns_prometheus_metrics_text():
    url = "http://127.0.0.1:8000/metrics"
    headers = {
        "Accept": "text/plain"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
        content_type = response.headers.get("Content-Type", "")
        assert "text/plain" in content_type or "text" in content_type, f"Expected Content-Type to include 'text/plain' but got '{content_type}'"
        # Basic check for Prometheus metric text format (presence of # HELP and # TYPE lines)
        content = response.text
        assert "# HELP" in content and "# TYPE" in content, "Response does not contain Prometheus metric format headers"
        assert len(content.strip()) > 0, "Metrics response is empty"
    except requests.RequestException as e:
        assert False, f"Request to /metrics failed: {e}"

test_get_metrics_returns_prometheus_metrics_text()