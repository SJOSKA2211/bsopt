import requests


def test_get_health_returns_healthy_status():
    url = "http://127.0.0.1:8000/health"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /health failed: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "status" in json_data, "Response JSON missing 'status' key"
    assert json_data["status"] == "healthy", f"Expected status 'healthy', got {json_data['status']}"


test_get_health_returns_healthy_status()
