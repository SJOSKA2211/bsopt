import pytest
import grpc
import asyncio
from unittest.mock import MagicMock
import time # For timestamping and simulating delays

# Mocking gRPC client and proto messages for the test environment
# In a real test setup, you would import generated clients and run a test gRPC server.

# Mock proto messages and services
class MockTokenResponse:
    def __init__(self, valid, user_id, email, tier, expires_at, issued_at, token_type, roles):
        self.valid = valid
        self.user_id = user_id
        self.email = email
        self.tier = tier
        self.expires_at = expires_at
        self.issued_at = issued_at
        self.token_type = token_type
        self.roles = roles

class MockTokenPairResponse:
    def __init__(self, access_token, refresh_token, token_type, expires_in, issued_at):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_type = token_type
        self.expires_in = expires_in
        self.issued_at = issued_at

class MockUserInfo:
    def __init__(self, user_id, email, full_name, tier, is_verified, mfa_enabled, created_at, last_login, roles, metadata):
        self.user_id = user_id
        self.email = email
        self.full_name = full_name
        self.tier = tier
        self.is_verified = is_verified
        self.mfa_enabled = mfa_enabled
        self.created_at = created_at
        self.last_login = last_login
        self.roles = roles
        self.metadata = metadata

class MockAPIKeyResponse: pass
class MockIntrospectionResponse: 
    def __init__(self, active, sub=None, username=None, token_type=None, exp=0, iat=0, scope=None, iss=None):
        self.active = active
        self.sub = sub
        self.username = username
        self.token_type = token_type
        self.exp = exp
        self.iat = iat
        self.scope = scope
        self.iss = iss

class MockTokenRequest:
    def __init__(self, token):
        self.token = token

class MockRevokeTokenRequest:
    def __init__(self, token):
        self.token = token
        
class MockCreateTokenPairRequest:
    def __init__(self, user_id, email, tier, scopes):
        self.user_id = user_id
        self.email = email
        self.tier = tier
        self.scopes = scopes
        
class MockRefreshTokenRequest:
    def __init__(self, refresh_token):
        self.refresh_token = refresh_token

# Mock gRPC channel and server components
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
    server.servicer_registry['AuthService'] = servicer

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
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await server.start()
    # In tests, we don't need to wait for termination, just ensure methods can be called.
    # await server.wait_for_termination() 

# Mocking asyncio.run for simplicity in testing servicer methods directly
def mock_asyncio_run(coro):
    # This bypasses actual async execution for direct method calls in tests
    # In real tests, you'd use pytest-asyncio or similar.
    # For this mock, we'll assume the coro completes.
    print("Mock asyncio.run called for a coroutine.")
    # You might want to run the coroutine in a test loop if needed.
    # For now, just acknowledge it's called.
    asyncio.get_event_loop().run_until_complete(coro)


# Monkey patch original modules/functions for testing
original_grpc = grpc
original_asyncio_run = asyncio.run
original_grpc_aio_server = grpc.aio.server
original_grpc_aio_secure_port = grpc.aio.server.add_secure_port
original_grpc_composite_channel_credentials = grpc.composite_channel_credentials
original_grpc_ssl_channel_credentials = grpc.ssl_channel_credentials
original_grpc_ssl_server_credentials = grpc.ssl_server_credentials
original_add_auth_service_servicer = auth_pb2_grpc.add_AuthServiceServicer_to_server
original_add_health_servicer = health_pb2_grpc.add_HealthServicer_to_server
original_health_servicer = health.HealthServicer
original_auth_revoke_token = auth.revoke_token # Mock auth functions too

@pytest.fixture(scope="module", autouse=True)
def mock_grpc_environment():
    """Mocks gRPC components and auth functions for testing."""
    grpc.aio.server = MockGrpcServer
    grpc.aio.server.add_secure_port = mock_grpc_aio_server
    grpc.composite_channel_credentials = mock_composite_channel_credentials
    grpc.ssl_channel_credentials = mock_ssl_channel_credentials
    grpc.ssl_server_credentials = mock_ssl_server_credentials
    auth_pb2_grpc.add_AuthServiceServicer_to_server = add_AuthServiceServicer_to_server
    health_pb2_grpc.add_HealthServicer_to_server = add_HealthServicer_to_server
    health.HealthServicer = MockHealthServicer
    auth.revoke_token = lambda token: REVOKED_TOKENS.add(token) # Mock revoke_token
    asyncio.run = mock_asyncio_run
    
    # Mocking the global REVOKED_TOKENS set used by auth.py and servicer
    global REVOKED_TOKENS
    REVOKED_TOKENS = set()

    yield

    # Teardown mocks
    grpc.aio.server = original_grpc_aio_server
    grpc.aio.server.add_secure_port = original_grpc_aio_secure_port
    grpc.composite_channel_credentials = original_grpc_composite_channel_credentials
    grpc.ssl_channel_credentials = original_grpc_ssl_channel_credentials
    grpc.ssl_server_credentials = original_grpc_ssl_server_credentials
    auth_pb2_grpc.add_AuthServiceServicer_to_server = original_add_auth_service_servicer
    health_pb2_grpc.add_HealthServicer_to_server = original_add_health_servicer
    health.HealthServicer = original_health_servicer
    auth.revoke_token = original_auth_revoke_token # Restore original revoke_token
    asyncio.run = original_asyncio_run

@pytest.fixture(scope="module")
def auth_service_servicer():
    """Provides a mock AuthServicer instance."""
    # This fixture sets up the mock server and servicer for tests
    mock_server = MockGrpcServer()
    mock_servicer = MockAuthServiceStub(None)
    add_AuthServiceServicer_to_server(mock_servicer, mock_server)
    # Mocking serve() to not block indefinitely in tests
    auth_pb2_grpc.AuthService.Serve = mock_serve 
    return mock_servicer

@pytest.mark.asyncio
async def test_validate_token_valid(auth_service_servicer: MockAuthServiceStub):
    """Tests validating a valid token."""
    mock_request = MockTokenRequest(token="valid-token-abc")
    mock_context = MagicMock()

    response = await auth_service_servicer.ValidateToken(mock_request)
    
    assert response.valid is True
    assert response.user_id == "test-user-123"
    assert response.email == "test@example.com"
    assert response.tier == "premium"

@pytest.mark.asyncio
async def test_validate_token_invalid(auth_service_servicer: MockAuthServiceStub):
    """Tests validating an invalid token."""
    mock_request = MockTokenRequest(token="invalid-token")
    mock_context = MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(mock_request)
    
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "Token is invalid or expired" in str(excinfo.value.details())

@pytest.mark.asyncio
async def test_validate_token_revoked(auth_service_servicer: MockAuthServiceStub):
    """Tests validating a revoked token."""
    mock_request = MockTokenRequest(token="revoked-token-xyz")
    mock_context = MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(mock_request)
    
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "Token has been revoked" in str(excinfo.value.details())

@pytest.mark.asyncio
async def test_create_token_pair(auth_service_servicer: MockAuthServiceStub):
    """Tests creating a token pair."""
    mock_request = auth_pb2.CreateTokenRequest(
        user_id="user-for-tokens",
        email="user@example.com",
        tier="basic",
        scopes=["read:data"]
    )
    mock_context = MagicMock()

    response = await auth_service_servicer.CreateTokenPair(mock_request)
    
    assert response.access_token is not None
    assert response.refresh_token is not None
    assert response.token_type == "Bearer"
    assert response.expires_in > 0

@pytest.mark.asyncio
async def test_refresh_token(auth_service_servicer: MockAuthServiceStub):
    """Tests refreshing a token."""
    mock_request = auth_pb2.RefreshTokenRequest(refresh_token="valid-refresh-token")
    mock_context = MagicMock()

    response = await auth_service_servicer.RefreshToken(mock_request)
    
    assert response.valid is True
    assert response.token_type == "access"
    assert response.access_token is not None
    assert response.expires_in > 0

@pytest.mark.asyncio
async def test_revoke_token(auth_service_servicer: MockAuthServiceStub):
    """Tests revoking a token."""
    token_to_revoke = "token-to-revoke-123"
    # Simulate adding to revoked set (done within auth.py)
    auth.revoke_token(token_to_revoke) 

    mock_request = auth_pb2.RevokeTokenRequest(token=token_to_revoke)
    mock_context = MagicMock()

    response = await auth_service_servicer.RevokeToken(mock_request)
    
    assert response == empty_pb2.Empty() 

    # Verify revocation by attempting to validate the revoked token
    invalid_request = auth_pb2.TokenRequest(token=token_to_revoke)
    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.ValidateToken(invalid_request)
    
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "Token has been revoked" in str(excinfo.value.details())

@pytest.mark.asyncio
async def test_get_user_info(auth_service_servicer: MockAuthServiceStub):
    """Tests retrieving user info via token."""
    mock_request = auth_pb2.GetUserInfoRequest(token="valid-token-abc")
    mock_context = MagicMock()

    response = await auth_service_servicer.GetUserInfo(mock_request)
    
    assert response.user_id == "test-user-123"
    assert response.email == "test@example.com"
    assert response.tier == "premium"
    assert response.full_name == "User Full Name Placeholder" 
    assert response.roles == ["user"] 

@pytest.mark.asyncio
async def test_get_user_info_invalid_token(auth_service_servicer: MockAuthServiceStub):
    """Tests getting user info with an invalid token."""
    mock_request = auth_pb2.GetUserInfoRequest(token="invalid-token-for-userinfo")
    mock_context = MagicMock()

    with pytest.raises(grpc.RpcError) as excinfo:
        await auth_service_servicer.GetUserInfo(mock_request)
    
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert "Token is invalid or expired" in str(excinfo.value.details())

# Note: IntrospectToken and ValidateAPIKey are UNIMPLEMENTED and thus not tested here.
