import requests

def test_get_health_returns_healthy_status():
    url = "http://127.0.0.1:8000/health"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"
    assert data == {"status": "healthy"}, f"Expected JSON {{'status': 'healthy'}}, got {data}"

test_get_health_returns_healthy_status()