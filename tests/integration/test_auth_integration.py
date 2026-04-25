import os
import uuid

import grpc
import pytest

from src.shared.protos import auth_pb2, auth_pb2_grpc

@pytest.mark.asyncio
async def test_auth_grpc_mtls_connection():
    """
    Validates that the mTLS connection between a client and the Auth server works.
    Requires the Auth service to be running with certs in .pki.
    """
    addr = os.getenv("AUTH_SVC_ADDR", "localhost:50051")
    ca_cert_path = ".pki/root_ca.crt"
    client_cert_path = ".pki/api_service.crt"
    client_key_path = ".pki/api_service.key"

    if not all(os.path.exists(p) for p in [ca_cert_path, client_cert_path, client_key_path]): # noqa: ASYNC240
        pytest.skip("mTLS certificates not found in .pki")

    with open(ca_cert_path, "rb") as f: # noqa: ASYNC230
        root_certs = f.read()
    with open(client_cert_path, "rb") as f: # noqa: ASYNC230
        cert_chain = f.read()
    with open(client_key_path, "rb") as f: # noqa: ASYNC230
        private_key = f.read()

    creds = grpc.ssl_channel_credentials(
        root_certificates=root_certs,
        private_key=private_key,
        certificate_chain=cert_chain
    )

    try:
        async with grpc.aio.secure_channel(addr, creds) as channel:
            stub = auth_pb2_grpc.AuthServiceStub(channel)
            # Just a simple health check or any method
            response = await stub.ValidateToken(
                auth_pb2.TokenRequest(token="invalid-test-token"),
                timeout=5
            )
            # We expect a response even if token is invalid, 
            # as long as the TLS handshake succeeds and the logic executes.
            assert not response.valid
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            # TLS Handshake succeeded, but token logic failed (Expected for dummy token)
            pass
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            pytest.fail(f"Auth service at {addr} is unavailable or TLS handshake failed")
        else:
            pytest.fail(f"gRPC Error: {e.code()} - {e.details()}")

@pytest.mark.asyncio
async def test_auth_grpc_token_lifecycle():
    """
    Tests CreateTokenPair and then ValidateToken.
    """
    addr = os.getenv("AUTH_SVC_ADDR", "localhost:50051")
    ca_cert_path = ".pki/root_ca.crt"
    client_cert_path = ".pki/api_service.crt"
    client_key_path = ".pki/api_service.key"

    if not all(os.path.exists(p) for p in [ca_cert_path, client_cert_path, client_key_path]): # noqa: ASYNC240
        pytest.skip("mTLS certificates not found in .pki")

    with open(ca_cert_path, "rb") as f: # noqa: ASYNC230
        root_certs = f.read()
    with open(client_cert_path, "rb") as f: # noqa: ASYNC230
        cert_chain = f.read()
    with open(client_key_path, "rb") as f: # noqa: ASYNC230
        private_key = f.read()

    creds = grpc.ssl_channel_credentials(root_certs, private_key, cert_chain)

    async with grpc.aio.secure_channel(addr, creds) as channel:
        stub = auth_pb2_grpc.AuthServiceStub(channel)
        
        user_id = str(uuid.uuid4())
        email = "test@example.com"
        
        # 1. Create tokens
        create_resp = await stub.CreateTokenPair(
            auth_pb2.CreateTokenRequest(
                user_id=user_id,
                email=email,
                tier="pro",
                scopes=["read", "write"]
            )
        )
        assert create_resp.access_token
        assert create_resp.refresh_token
        
        # 2. Validate token
        val_resp = await stub.ValidateToken(
            auth_pb2.TokenRequest(token=create_resp.access_token)
        )
        assert val_resp.valid
        assert val_resp.user_id == user_id
        assert val_resp.email == email
        assert "read" in val_resp.roles
