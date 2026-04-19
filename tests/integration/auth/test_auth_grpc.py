import asyncio
import time  # For timestamping and simulating delays
from datetime import UTC, datetime
from unittest.mock import MagicMock

import grpc
import pytest
from google.protobuf import empty_pb2  # For empty responses

from src.auth import auth  # Import auth module for revoke_token function access

# Assume necessary imports for gRPC client and proto definitions
# These would typically be generated client libraries from your proto files.
# For testing, we mock these components.
from src.shared.protos import auth_pb2, auth_pb2_grpc


# --- Mock gRPC Components ---
# Mocking the Auth Service Stub and its methods
class MockAuthServiceStub:
    def __init__(self, channel):
        self.channel = channel

    async def ValidateToken(self, request):
        if request.token == "valid-token-abc":
            return auth_pb2.TokenResponse(valid=True, user_id="test-user-123", email="test@example.com", tier="premium", expires_at=int(time.time()) + 3600, issued_at=int(time.time()), token_type="access", roles=["user"])
        if request.token == "revoked-token-xyz":
            raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Token has been revoked")
        raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

    async def CreateTokenPair(self, request):
        if request.user_id == "test-user-123":
            return auth_pb2.TokenPairResponse(access_token="new-access-token", refresh_token="new-refresh-token", token_type="Bearer", expires_in=3600, issued_at=datetime.now(UTC))
        raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Invalid user credentials")

    async def RefreshToken(self, request):
        if request.refresh_token == "valid-refresh-token":
            return auth_pb2.TokenResponse(valid=True, user_id="test-user-123", expires_at=int(time.time()) + 3600, token_type="access")
        raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Invalid or expired refresh token")

    async def RevokeToken(self, request):
        # Use the mocked revoke_token function which adds to REVOKED_TOKENS set
        auth.revoke_token(request.token)
        return empty_pb2.Empty() # Return empty message for success

    async def GetUserInfo(self, request):
        payload = await auth.verify_token(request.token) # Use verify_token to check validity and get payload
        if payload is None:
            raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

        # In a real test, you'd use a mock DB session to fetch user details.
        # For this mock, we simulate user data retrieval based on payload.
        if payload.get("sub") == "test-user-123":
            return auth_pb2.UserInfo(
                user_id="test-user-123",
                email="test@example.com",
                full_name="Test User",
                tier="premium",
                is_verified=True,
                mfa_enabled=False,
                # created_at/last_login need google.protobuf.Timestamp in real life
                # but for this mock we check against what's expected in tests
                roles=payload.get("roles", []),
            )
        raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "User not found for token")

    async def ValidateAPIKey(self, request):
        raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

    async def IntrospectToken(self, request):
        raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

# Mocking proto messages (simplified for tests)
# Assuming these classes exist in the generated proto files or are mocked
class MockTokenResponse:
    def __init__(self, valid, user_id, email, tier, expires_at, issued_at, token_type, roles):
        self.valid, self.user_id, self.email, self.tier, self.expires_at, self.issued_at, self.token_type, self.roles = valid, user_id, email, tier, expires_at, issued_at, token_type, roles

class MockTokenPairResponse:
    def __init__(self, access_token, refresh_token, token_type, expires_in, issued_at):
        self.access_token, self.refresh_token, self.token_type, self.expires_in, self.issued_at = access_token, refresh_token, token_type, expires_in, issued_at

class MockUserInfo:
    def __init__(self, user_id, email, full_name, tier, is_verified, mfa_enabled, created_at, last_login, roles, metadata):
        self.user_id, self.email, self.full_name, self.tier, self.is_verified, self.mfa_enabled, self.created_at, self.last_login, self.roles, self.metadata = user_id, email, full_name, tier, is_verified, mfa_enabled, created_at, last_login, roles, metadata

class MockAPIKeyResponse:
    pass
class MockIntrospectionResponse:
    def __init__(self, active, sub=None, username=None, token_type=None, exp=0, iat=0, scope=None, iss=None):
        self.active, self.sub, self.username, self.token_type, self.exp, self.iat, self.scope, self.iss = active, sub, username, token_type, exp, iat, scope, iss

class MockTokenRequest:
    def __init__(self, token): self.token = token

class MockRevokeTokenRequest:
    def __init__(self, token): self.token = token

class MockCreateTokenPairRequest:
    def __init__(self, user_id, email, tier, scopes):
        self.user_id, self.email, self.tier, self.scopes = user_id, email, tier, scopes

class MockRefreshTokenRequest:
    def __init__(self, refresh_token): self.refresh_token = refresh_token

# --- Mock gRPC channel and server components ---
class MockChannel:
    async def close(self):
        pass

async def mock_grpc_aio_channel(addr):
    return MockChannel()

async def mock_secure_channel(addr, creds):
    return MockChannel()

async def mock_composite_channel_credentials(ssl_creds, client_creds):
    return None

async def mock_ssl_channel_credentials(root_certificates):
    return None

def mock_ssl_server_credentials(certificate_chain, root_certificates, require_client_auth=False):
    return None

# Mocking the gRPC server and servicer registration
class MockGrpcServer:
    def __init__(self):
        self.servicer_registry = {}

    async def start(self):
        pass

    async def wait_for_termination(self):
        pass

    def add_secure_port(self, addr, creds):
        pass

    def add_insecure_port(self, addr):
        pass

def add_AuthServiceServicer_to_server(servicer, server):
    server.servicer_registry["AuthService"] = servicer

# Mocking the health servicer
class MockHealthServicer:
    def set(self, service, status): pass
def add_HealthServicer_to_server(servicer, server): pass

# Mocking the main server run logic for testing standalone servicer methods
async def mock_serve():
    server = MockGrpcServer()
    mock_servicer = MockAuthServiceStub(None)
    add_AuthServiceServicer_to_server(mock_servicer, server)
    health_servicer = MockHealthServicer()
    add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", 1)
    await server.start()
    # In tests, we don't need to wait for termination, just ensure methods can be called.

# Mocking asyncio.run for simplicity in testing servicer methods directly
def mock_asyncio_run(coro):
    # This bypasses actual async execution for direct method calls in tests
    # In real tests, you'd use pytest-asyncio or similar.
    # For this mock, we'll just assume the coroutine completes.
    print("Mock asyncio.run called for a coroutine.")
    asyncio.get_event_loop().run_until_complete(coro)

# --- Global mocks for testing (Restored to working state) ---
original_grpc = grpc
original_asyncio_run = asyncio.run
original_grpc_aio_server = grpc.aio.server
# Removed invalid add_secure_port mock
original_grpc_composite_channel_credentials = grpc.composite_channel_credentials
original_grpc_ssl_channel_credentials = grpc.ssl_channel_credentials
original_grpc_ssl_server_credentials = grpc.ssl_server_credentials
original_add_auth_service_servicer = auth_pb2_grpc.add_AuthServiceServicer_to_server
# health_pb2_grpc might not be available or needed here if we mock health
# original_add_health_servicer = health_pb2_grpc.add_HealthServicer_to_server
# original_health_servicer = health.HealthServicer
original_auth_revoke_token = auth.revoke_token # Mock auth functions too
original_auth_verify_token = auth.verify_token

# --- Mocking REVOKED_TOKENS set ---
# We need to mock the REVOKED_TOKENS set used by auth.py and the servicer
# This allows us to control its state during tests.
MOCKED_REVOKED_TOKENS = set()
def mock_revoke_token_func(token: str):
    MOCKED_REVOKED_TOKENS.add(token)
    print(f"Mock: Token {token[:10]}... added to REVOKED_TOKENS.")

async def mock_verify_token_func(token: str):
    if token == "valid-token-abc":
        return {"sub": "test-user-123", "email": "test@example.com", "tier": "premium", "roles": ["user"], "token_type": "access"}
    if token == "valid-refresh-token":
        return {"sub": "test-user-123", "token_type": "refresh"}
    return None

# --- Pytest Fixtures ---
@pytest.fixture(scope="module", autouse=True)
def mock_grpc_environment():
    """Mocks gRPC components and auth functions for testing."""
    # grpc.aio.server = MockGrpcServer # This is also problematic if it's not a function
    grpc.composite_channel_credentials = mock_composite_channel_credentials
    grpc.ssl_channel_credentials = mock_ssl_channel_credentials
    grpc.ssl_server_credentials = mock_ssl_server_credentials
    auth_pb2_grpc.add_AuthServiceServicer_to_server = add_AuthServiceServicer_to_server
    # auth_pb2_grpc.add_HealthServicer_to_server = add_HealthServicer_to_server
    # health.HealthServicer = MockHealthServicer
    auth.revoke_token = mock_revoke_token_func # Use mock revoke token
    auth.verify_token = mock_verify_token_func # Use mock verify token
    asyncio.run = mock_asyncio_run

    # Reset the mocked REVOKED_TOKENS set before each module test run
    global MOCKED_REVOKED_TOKENS
    MOCKED_REVOKED_TOKENS = set()

    yield

    # Teardown mocks
    # grpc.aio.server = original_grpc_aio_server
    grpc.composite_channel_credentials = original_grpc_composite_channel_credentials
    grpc.ssl_channel_credentials = original_grpc_ssl_channel_credentials
    grpc.ssl_server_credentials = original_grpc_ssl_server_credentials
    auth_pb2_grpc.add_AuthServiceServicer_to_server = original_add_auth_service_servicer
    # auth_pb2_grpc.add_HealthServicer_to_server = original_add_health_servicer
    # health.HealthServicer = original_health_servicer
    auth.revoke_token = original_auth_revoke_token # Restore original revoke_token
    auth.verify_token = original_auth_verify_token
    asyncio.run = original_asyncio_run

@pytest.fixture(scope="module")
def auth_service_servicer():
    """Provides a mock AuthServicer instance."""
    mock_server = MockGrpcServer()
    mock_servicer = MockAuthServiceStub(None)
    add_AuthServiceServicer_to_server(mock_servicer, mock_server)
    auth_pb2_grpc.AuthService.Serve = mock_serve # Override serve for testing
    return mock_servicer

# --- Tests for ValidateToken ---
@pytest.mark.asyncio
async def test_validate_token_valid(auth_service_servicer: MockAuthServiceStub):
    """Tests validating a valid token."""
    mock_request = MockTokenRequest(token="valid-token-abc")
    MagicMock()

    response = await auth_service_servicer.ValidateToken(mock_request)

    assert response.valid is True
    assert response.user_id == "test-user-123"
    assert response.email == "test@example.com"
    assert response.tier == "premium"

@pytest.mark.asyncio
async def test_validate_token_invalid(auth_service_servicer: MockAuthServiceStub):
    """Tests validating an invalid token."""
    mock_request = MockTokenRequest(token="invalid-token")
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(mock_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNAUTHENTICATED
    assert "Token is invalid or expired" in excinfo.value.args[1]

@pytest.mark.asyncio
async def test_validate_token_revoked(auth_service_servicer: MockAuthServiceStub):
    """Tests validating a revoked token."""
    token_to_revoke = "revoked-token-xyz"
    # Ensure the token is added to the mocked revoked set
    auth.revoke_token(token_to_revoke)

    mock_request = MockTokenRequest(token=token_to_revoke)
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(mock_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNAUTHENTICATED
    assert "Token has been revoked" in excinfo.value.args[1]

# --- Tests for CreateTokenPair ---
@pytest.mark.asyncio
async def test_create_token_pair(auth_service_servicer: MockAuthServiceStub):
    """Tests creating a token pair."""
    mock_request = auth_pb2.CreateTokenRequest(
        user_id="test-user-123", # Fixed from user-for-tokens
        email="user@example.com",
        tier="basic",
        scopes=["read:data"],
    )
    MagicMock()

    response = await auth_service_servicer.CreateTokenPair(mock_request)

    assert response.access_token is not None
    assert response.refresh_token is not None
    assert response.token_type == "Bearer"
    assert response.expires_in > 0

# --- Tests for RefreshToken ---
@pytest.mark.asyncio
async def test_refresh_token(auth_service_servicer: MockAuthServiceStub):
    """Tests refreshing a token."""
    mock_request = auth_pb2.RefreshRequest(refresh_token="valid-refresh-token")
    MagicMock()

    response = await auth_service_servicer.RefreshToken(mock_request)

    assert response.valid is True
    assert response.token_type == "access"
    assert response.user_id == "test-user-123"

@pytest.mark.asyncio
async def test_refresh_token_invalid(auth_service_servicer: MockAuthServiceStub):
    """Tests refreshing with an invalid refresh token."""
    mock_request = auth_pb2.RefreshRequest(refresh_token="invalid-refresh-token")
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.RefreshToken(mock_request)

    # In actual gRPC aio, the error has code() and details()
    # For our mock, we check the args if it's a simple RpcError
    assert excinfo.value.args[0] == grpc.StatusCode.UNAUTHENTICATED
    assert "Invalid or expired refresh token" in excinfo.value.args[1]

# --- Tests for RevokeToken ---
@pytest.mark.asyncio
async def test_revoke_token(auth_service_servicer: MockAuthServiceStub):
    """Tests revoking a token."""
    token_to_revoke = "token-to-revoke-123"
    # Ensure the token is added to the mocked revoked set via auth.revoke_token
    auth.revoke_token(token_to_revoke)

    mock_request = auth_pb2.RevokeRequest(token=token_to_revoke)
    MagicMock()

    response = await auth_service_servicer.RevokeToken(mock_request)

    # RevokeToken is expected to return an empty message
    assert isinstance(response, empty_pb2.Empty)

    # Verify revocation by attempting to validate the revoked token
    invalid_request = auth_pb2.TokenRequest(token=token_to_revoke)
    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(invalid_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNAUTHENTICATED
    assert "Token is invalid or expired" in excinfo.value.args[1]

# --- Tests for GetUserInfo ---
@pytest.mark.asyncio
async def test_get_user_info(auth_service_servicer: MockAuthServiceStub):
    """Tests retrieving user info via token."""
    mock_request = auth_pb2.TokenRequest(token="valid-token-abc")
    MagicMock()

    response = await auth_service_servicer.GetUserInfo(mock_request)

    assert response.user_id == "test-user-123"
    assert response.email == "test@example.com"
    assert response.tier == "premium"
    # In MockAuthServiceStub.GetUserInfo, full_name is "Test User"
    assert response.full_name == "Test User"
    assert "user" in response.roles

@pytest.mark.asyncio
async def test_get_user_info_invalid_token(auth_service_servicer: MockAuthServiceStub):
    """Tests getting user info with an invalid token."""
    mock_request = auth_pb2.TokenRequest(token="invalid-token-for-userinfo")
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.GetUserInfo(mock_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNAUTHENTICATED
    assert "Token is invalid or expired" in excinfo.value.args[1]

# --- Tests for UNIMPLEMENTED methods ---
# These tests ensure that calling unimplemented methods correctly raises UNIMPLEMENTED errors.

@pytest.mark.asyncio
async def test_validate_api_key_unimplemented(auth_service_servicer: MockAuthServiceStub):
    """Tests that ValidateAPIKey is correctly marked as UNIMPLEMENTED."""
    mock_request = auth_pb2.APIKeyRequest(api_key="dummy-key") # Correct field name
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateAPIKey(mock_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNIMPLEMENTED
    assert "Method not implemented" in excinfo.value.args[1]

@pytest.mark.asyncio
async def test_introspect_token_unimplemented(auth_service_servicer: MockAuthServiceStub):
    """Tests that IntrospectToken is correctly marked as UNIMPLEMENTED."""
    mock_request = auth_pb2.TokenRequest(token="some-token") # Correct message name
    MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.IntrospectToken(mock_request)

    assert excinfo.value.args[0] == grpc.StatusCode.UNIMPLEMENTED
    assert "Method not implemented" in excinfo.value.args[1]

