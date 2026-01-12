from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from src.shared.security import verify_mtls, opa_authorize
from unittest.mock import patch

app = FastAPI()

@app.get("/secure-data", dependencies=[Depends(verify_mtls), Depends(opa_authorize("read", "secure_resource"))])
async def secure_data():
    return {"data": "secret"}

client = TestClient(app)

def test_mtls_and_opa_success():
    """Test successful mTLS and OPA authorization."""
    with patch("src.shared.security.OPAEnforcer.is_authorized", return_value=True):
        headers = {
            "X-SSL-Client-Verify": "SUCCESS",
            "X-SSL-Client-S-DN": "CN=service-a",
            "X-User-Id": "user1",
            "X-User-Role": "admin"
        }
        response = client.get("/secure-data", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"data": "secret"}

def test_mtls_failure():
    """Test mTLS verification failure."""
    headers = {
        "X-SSL-Client-Verify": "NONE",
        "X-User-Id": "user1",
        "X-User-Role": "admin"
    }
    response = client.get("/secure-data", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "mTLS verification failed"

def test_opa_failure():
    """Test OPA authorization failure."""
    with patch("src.shared.security.OPAEnforcer.is_authorized", return_value=False):
        headers = {
            "X-SSL-Client-Verify": "SUCCESS",
            "X-SSL-Client-S-DN": "CN=service-a",
            "X-User-Id": "user1",
            "X-User-Role": "guest"
        }
        response = client.get("/secure-data", headers=headers)
        assert response.status_code == 403
        assert "OPA Authorization failed" in response.json()["detail"]

def test_missing_headers():
    """Test failure when headers are missing."""
    response = client.get("/secure-data")
    assert response.status_code == 403
