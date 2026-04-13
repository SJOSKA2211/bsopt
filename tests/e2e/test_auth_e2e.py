import asyncio
import os
import pytest
import httpx
import grpc
from datetime import UTC, datetime
from src.shared.protos import auth_pb2, auth_pb2_grpc

# Configuration for E2E tests hitting the live network
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
AUTH_GRPC_URL = os.getenv("AUTH_SERVICE_GRPC_URL", "localhost:50051")

@pytest.fixture
async def api_client():
    async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
        yield client

@pytest.fixture
async def auth_grpc_stub():
    # Use insecure for now if running locally, or handle certs if needed
    SECURE = os.getenv("GRPC_SECURE", "false").lower() == "true"
    if SECURE:
        # In a real E2E environment, we'd load the CA cert
        ca_cert = "/etc/pki/root_ca.crt"
        if os.path.exists(ca_cert):
            with open(ca_cert, "rb") as f:
                creds = grpc.ssl_channel_credentials(f.read())
            channel = grpc.aio.secure_channel(AUTH_GRPC_URL, creds)
        else:
            channel = grpc.aio.insecure_channel(AUTH_GRPC_URL)
    else:
        channel = grpc.aio.insecure_channel(AUTH_GRPC_URL)
        
    stub = auth_pb2_grpc.AuthServiceStub(channel)
    yield stub
    await channel.close()

@pytest.mark.asyncio
async def test_auth_full_lifecycle_e2e(api_client, auth_grpc_stub):
    """
    True Integration: Register -> Login -> Validate via gRPC -> Refresh -> Me.
    NO MOCKS ALLOWED.
    """
    email = f"e2e_test_{int(datetime.now(UTC).timestamp())}@manifold.test"
    password = "StrongPassword123!"
    
    # 1. Register
    reg_resp = await api_client.post("/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "E2E Test User"
    })
    assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"
    
    # 2. Login
    login_resp = await api_client.post("/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    data = login_resp.json()["data"]
    access_token = data["access_token"]
    refresh_token = data["refresh_token"]
    
    # 3. Validate Token via gRPC (Internal Service Check)
    grpc_req = auth_pb2.TokenRequest(token=access_token)
    grpc_resp = await auth_grpc_stub.ValidateToken(grpc_req)
    assert grpc_resp.valid is True
    assert grpc_resp.email == email
    
    # 4. Access Protected Route (/me)
    me_resp = await api_client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["data"]["email"] == email
    
    # 5. Refresh Token
    refresh_resp = await api_client.post("/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert refresh_resp.status_code == 200
    new_data = refresh_resp.json()["data"]
    assert new_data["access_token"] != access_token
    
    # 6. Logout (Revoke)
    logout_resp = await api_client.post("/auth/logout", headers={"Authorization": f"Bearer {access_token}"})
    assert logout_resp.status_code == 200
    
    # 7. Validate via gRPC again (Should be invalid/revoked)
    revoked_resp = await auth_grpc_stub.ValidateToken(auth_pb2.TokenRequest(token=access_token))
    assert revoked_resp.valid is False

@pytest.mark.asyncio
async def test_auth_invalid_login(api_client):
    """Ensure invalid credentials fail without mocks."""
    login_resp = await api_client.post("/auth/login", json={
        "email": "nonexistent@manifold.test",
        "password": "WrongPassword!"
    })
    assert login_resp.status_code == 401
