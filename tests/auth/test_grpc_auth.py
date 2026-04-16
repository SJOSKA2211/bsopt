from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.protobuf import empty_pb2
from jwt.exceptions import ExpiredSignatureError

# Import the servicer and its dependencies
from src.auth.grpc_server import AuthServicer
from src.protos import auth_pb2


# Mock User model from src.database.models
class MockUser:
    def __init__(self, id, email, tier, is_verified=True, mfa_enabled=False, full_name="", created_at=None, last_login_at=None):
        self.id = id
        self.email = email
        self.tier = tier
        self.is_verified = is_verified
        self.mfa_enabled = mfa_enabled
        self.full_name = full_name
        self.created_at = created_at or datetime.now(UTC)
        self.last_login_at = last_login_at

# Mock TokenData structure
class TokenData:
    def __init__(self, user_id, email, tier, token_type, exp, iat, jti=None, scopes=None):
        self.user_id = user_id
        self.email = email
        self.tier = tier
        self.token_type = token_type
        self.exp = exp
        self.iat = iat
        self.jti = jti
        self.scopes = scopes if scopes is not None else []

# --- Fixtures ---

@pytest.fixture
def mock_auth_service():
    """Mocks the auth_service instance used by the gRPC server."""
    with patch("src.auth.grpc_server.auth_service") as mock_svc:
        mock_svc.validate_token = AsyncMock()
        mock_svc.revoke_token = AsyncMock()
        mock_svc.create_token_pair = MagicMock() # This is sync in the server
        mock_svc.decode_token = MagicMock()
        yield mock_svc

@pytest.fixture
def mock_db_manager():
    """Mocks the db_manager and its async_session_factory."""
    with patch("src.auth.grpc_server.db_manager") as mock_db_mgr:
        mock_db_session = AsyncMock()
        mock_db_mgr.async_session_factory.return_value.__aenter__.return_value = mock_db_session
        yield mock_db_session

@pytest.fixture
def mock_centralized_cache_service():
    """Mocks the centralized_cache_service object."""
    with patch("src.auth.grpc_server.centralized_cache_service") as mock_cache:
        mock_cache.get_user_cached = AsyncMock(return_value=None)
        mock_cache.set_user_cached = AsyncMock()
        mock_cache.get_api_key_cached = AsyncMock(return_value=None)
        mock_cache.set_api_key_cached = AsyncMock()
        mock_cache.update_api_key_last_used = AsyncMock()
        yield mock_cache

@pytest.fixture
def mock_grpc_context():
    """Mocks the grpc.aio.ServicerContext object."""
    mock_context = MagicMock(spec=grpc.aio.ServicerContext)
    mock_context.set_code.return_value = None
    mock_context.set_details.return_value = None
    return mock_context

@pytest.fixture
def auth_servicer():
    """Provides an instance of the AuthServicer."""
    return AuthServicer()

# --- Mock Data ---

def create_mock_user(user_id="user-123", email="test@example.com", tier="free", **kwargs):
    return MockUser(id=user_id, email=email, tier=tier, **kwargs)

def create_mock_token_data(user_id="user-123", email="test@example.com", tier="free", token_type="access", **kwargs):
    now = datetime.now(UTC)
    return TokenData(
        user_id=user_id,
        email=email,
        tier=tier,
        token_type=token_type,
        exp=kwargs.get("exp", now + timedelta(hours=1)),
        iat=kwargs.get("iat", now - timedelta(minutes=1)),
        scopes=kwargs.get("scopes", ["read"]),
        jti=kwargs.get("jti", "mock-jti")
    )

# --- Test Cases ---

@pytest.mark.asyncio
async def test_validate_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token validation with a valid token."""
    mock_token_data = create_mock_token_data(
        user_id="test-user-id",
        email="test@example.com",
        tier="free",
        token_type="access",
        scopes=["read"],
        jti="valid-jti"
    )
    mock_auth_service.validate_token.return_value = mock_token_data

    mock_request = MagicMock()
    mock_request.token = "valid.jwt.token.for.success"

    response = await auth_servicer.ValidateToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.jwt.token.for.success")
    assert response.valid is True
    assert response.user_id == "test-user-id"
    assert response.token_type == "access"
    mock_grpc_context.set_code.assert_not_called()

@pytest.mark.asyncio
async def test_validate_token_expired(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests token validation when the token has expired."""
    mock_auth_service.validate_token.side_effect = ExpiredSignatureError("Token expired")

    mock_request = MagicMock()
    mock_request.token = "expired.jwt.token"

    response = await auth_servicer.ValidateToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("expired.jwt.token")
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED)
    mock_grpc_context.set_details.assert_called_once_with("Token has expired")

@pytest.mark.asyncio
async def test_refresh_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token refresh."""
    valid_refresh_token_data = create_mock_token_data(token_type="refresh", jti="refresh-jti")
    new_access_token_data = create_mock_token_data(token_type="access", jti="new-access-jti")

    mock_auth_service.validate_token.return_value = valid_refresh_token_data
    
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "new.access.token"
    mock_token_pair.refresh_token = "new.refresh.token"
    mock_token_pair.token_type = "access"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    mock_auth_service.decode_token.return_value = new_access_token_data

    mock_request = MagicMock()
    mock_request.refresh_token = "valid.refresh.token"

    response = await auth_servicer.RefreshToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.refresh.token")
    mock_auth_service.create_token_pair.assert_called_once_with(
        user_id=valid_refresh_token_data.user_id,
        email=valid_refresh_token_data.email,
        tier=valid_refresh_token_data.tier,
        scopes=valid_refresh_token_data.scopes
    )
    assert response.valid is True
    assert response.user_id == new_access_token_data.user_id
    assert response.token_type == "access"

@pytest.mark.asyncio
async def test_revoke_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token revocation."""
    mock_auth_service.revoke_token.return_value = None

    mock_request = MagicMock()
    mock_request.token = "token.to.revoke"

    response = await auth_servicer.RevokeToken(mock_request, mock_grpc_context)

    mock_auth_service.revoke_token.assert_called_once_with("token.to.revoke")
    assert isinstance(response, empty_pb2.Empty)

@pytest.mark.asyncio
async def test_get_user_info_success_from_cache(auth_servicer, mock_auth_service, mock_centralized_cache_service, mock_grpc_context):
    """Tests GetUserInfo when data is fetched from the distributed cache."""
    mock_token_data = create_mock_token_data(user_id="cache-user-id")
    mock_auth_service.validate_token.return_value = mock_token_data

    # Mock cache hit
    mock_cached_user_dict = {
        "user_id": "cache-user-id",
        "email": "cache@example.com",
        "tier": "enterprise",
        "full_name": "Cache User",
        "is_verified": True,
        "mfa_enabled": True,
        "created_at": "2020-09-13T12:26:40Z",
        "last_login": "2020-09-13T12:26:40Z",
        "roles": ["enterprise"]
    }
    mock_centralized_cache_service.get_user_cached.return_value = mock_cached_user_dict

    mock_request = MagicMock()
    mock_request.token = "valid.token.for.cacheuser"

    response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.token.for.cacheuser")
    mock_centralized_cache_service.get_user_cached.assert_called_once_with("cache-user-id")
    assert response.user_id == "cache-user-id"
    assert response.email == "cache@example.com"

@pytest.mark.asyncio
async def test_get_user_info_success_from_db(auth_servicer, mock_auth_service, mock_db_manager, mock_centralized_cache_service, mock_grpc_context):
    """Tests GetUserInfo when data is fetched from the database."""
    mock_token_data = create_mock_token_data(user_id="db-user-id")
    mock_auth_service.validate_token.return_value = mock_token_data
    mock_centralized_cache_service.get_user_cached.return_value = None

    # Mock DB interaction
    mock_user = create_mock_user(user_id="db-user-id", email="db@example.com", tier="premium", full_name="DB User")
    mock_db_session = mock_db_manager
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db_session.execute.return_value = mock_result

    mock_request = MagicMock()
    mock_request.token = "valid.token.for.dbuser"

    response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.token.for.dbuser")
    mock_centralized_cache_service.get_user_cached.assert_called_once_with("db-user-id")
    mock_db_session.execute.assert_called_once()
    mock_centralized_cache_service.set_user_cached.assert_called_once()
    assert response.user_id == "db-user-id"

@pytest.mark.asyncio
async def test_create_token_pair_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful creation of a token pair."""
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "new-access-token"
    mock_token_pair.refresh_token = "new-refresh-token"
    mock_token_pair.token_type = "Bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair

    mock_request = auth_pb2.CreateTokenRequest(
        user_id="user-to-create-tokens",
        email="user@example.com",
        tier="enterprise",
        scopes=["admin", "read"]
    )

    response = await auth_servicer.CreateTokenPair(mock_request, mock_grpc_context)

    mock_auth_service.create_token_pair.assert_called_once_with(
        user_id="user-to-create-tokens",
        email="user@example.com",
        tier="enterprise",
        scopes=["admin", "read"]
    )
    assert response.access_token == "new-access-token"
    assert response.refresh_token == "new-refresh-token"

@pytest.mark.asyncio
async def test_validate_api_key_success(auth_servicer, mock_centralized_cache_service, mock_grpc_context):
    """Tests successful API key validation via cache."""
    mock_api_key_resp_dict = {
        "valid": True,
        "user_id": "api-user-1",
        "email": "api@example.com",
        "tier": "enterprise",
        "key_name": "my-api-key",
        "created_at": "2020-09-13T12:26:40Z"
    }
    mock_centralized_cache_service.get_api_key_cached.return_value = mock_api_key_resp_dict

    mock_request = MagicMock()
    mock_request.api_key = "valid-api-key-secret"

    response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

    mock_centralized_cache_service.get_api_key_cached.assert_called_once()
    mock_centralized_cache_service.update_api_key_last_used.assert_called_once()
    assert response.valid is True
    assert response.user_id == "api-user-1"

@pytest.mark.asyncio
async def test_introspect_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token introspection."""
    mock_token_data = create_mock_token_data(
        user_id="intro-user-id",
        email="intro@example.com",
        tier="enterprise",
        token_type="bearer",
        scopes=["admin", "read"]
    )
    mock_auth_service.validate_token.return_value = mock_token_data

    mock_request = MagicMock()
    mock_request.token = "valid.token.for.introspection"

    response = await auth_servicer.IntrospectToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.token.for.introspection")
    assert response.active is True
    assert response.sub == "intro-user-id"
    assert response.username == "intro@example.com"
    assert response.scope == "admin read"
