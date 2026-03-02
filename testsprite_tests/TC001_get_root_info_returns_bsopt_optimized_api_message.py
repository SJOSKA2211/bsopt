import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_get_root_info_returns_bsopt_optimized_api_message():
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to root endpoint failed: {e}"

    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not a valid JSON"

    assert "message" in json_data, "Response JSON does not contain 'message' key"
    assert json_data["message"] == "BS-Opt Optimized API", f"Expected message 'BS-Opt Optimized API' but got '{json_data['message']}'"

test_get_root_info_returns_bsopt_optimized_api_message()