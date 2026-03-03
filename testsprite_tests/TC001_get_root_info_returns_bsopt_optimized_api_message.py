import requests


def test_get_root_info_returns_bsopt_optimized_api_message():
    url = "http://127.0.0.1:8000/"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "message" in json_data, "Response JSON does not contain 'message' key"
    assert json_data["message"] == "BS-Opt Optimized API", (
        f"Expected message 'BS-Opt Optimized API' but got {json_data['message']}"
    )


test_get_root_info_returns_bsopt_optimized_api_message()
