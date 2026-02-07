import requests

BASE_URL = "http://localhost:4000"
TIMEOUT = 30
HEADERS = {"Content-Type": "application/json"}


def test_authentication_endpoint_delegates_to_betterauth():
    session = requests.Session()
    session.headers.update(HEADERS)

    test_requests = [
        {
            "method": "POST",
            "path": "/api/auth/login",
            "json": {"username": "testuser", "password": "testpass"},
        },
        {"method": "POST", "path": "/api/auth/logout", "json": {}},
        {
            "method": "GET",
            "path": "/api/auth/callback",
            "params": {"code": "dummycode", "state": "dummystate"},
        },
        {
            "method": "POST",
            "path": "/api/auth/passwordless",
            "json": {"email": "test@example.com"},
        },
    ]

    for req in test_requests:
        url = BASE_URL + req["path"]
        try:
            if req["method"] == "GET":
                response = session.get(
                    url, params=req.get("params", None), timeout=TIMEOUT
                )
            elif req["method"] == "POST":
                response = session.post(
                    url, json=req.get("json", None), timeout=TIMEOUT
                )
            else:
                continue  # Skip unsupported method in this test context

            # Assert the request was delegated and successful (HTTP 200)
            assert (
                response.status_code == 200
            ), f"Expected HTTP 200 for {req['method']} {req['path']}, got {response.status_code} with body: {response.text}"
        except requests.RequestException as e:
            assert False, f"Request to {url} failed with exception: {e}"


test_authentication_endpoint_delegates_to_betterauth()
