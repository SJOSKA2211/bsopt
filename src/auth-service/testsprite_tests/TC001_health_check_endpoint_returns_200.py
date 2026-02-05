import requests


def test_health_check_returns_200():
    base_url = "http://localhost:4000"
    url = f"{base_url}/"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    except requests.RequestException as e:
        assert False, f"HTTP request failed: {e}"

test_health_check_returns_200()