import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest
import grpc
from google.protobuf import timestamp_pb2, empty_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

# Import the servicer and its dependencies
from src.auth.grpc_server import AuthServicer
from src.protos import auth_pb2
from jwt.exceptions import ExpiredSignatureError, PyJWTError

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

# Mock TokenData structure if not importable
try:
    from src.auth.core.tokens import TokenData
except ImportError:
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
        yield mock_svc

@pytest.fixture
def mock_db_manager():
    """Mocks the db_manager and its async_session_factory."""
    with patch("src.auth.grpc_server.db_manager") as mock_db_mgr:
        mock_db_session = MagicMock()
        mock_db_mgr.async_session_factory.return_value.__aenter__.return_value = mock_db_session
        yield mock_db_session

@pytest.fixture
def mock_db_cache():
    """Mocks the db_cache object."""
    with patch("src.auth.grpc_server.db_cache") as mock_cache:
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

def create_mock_user_info_proto(user_id="user-123", email="test@example.com", tier="free", **kwargs):
    created_at = kwargs.get("created_at", datetime.now(UTC))
    last_login_at = kwargs.get("last_login_at", datetime.now(UTC))

    ts_created_at = timestamp_pb2.Timestamp()
    ts_created_at.FromDatetime(created_at)
    ts_last_login = timestamp_pb2.Timestamp()
    ts_last_login.FromDatetime(last_login_at)

    return auth_pb2.UserInfo(
        user_id=user_id,
        email=email,
        tier=tier,
        full_name=kwargs.get("full_name", ""),
        is_verified=kwargs.get("is_verified", True),
        mfa_enabled=kwargs.get("mfa_enabled", False),
        created_at=ts_created_at,
        last_login=ts_last_login,
        roles=[tier], # Assuming roles are derived from tier for this test
        metadata=kwargs.get("metadata", {})
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
async def test_validate_token_invalid(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests token validation when the token is invalid (format/signature)."""
    mock_auth_service.validate_token.side_effect = PyJWTError("Invalid token")

    mock_request = MagicMock()
    mock_request.token = "invalid.jwt.token"

    response = await auth_servicer.ValidateToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("invalid.jwt.token")
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED)
    mock_grpc_context.set_details.assert_called_once_with("Invalid token")

@pytest.mark.asyncio
async def test_validate_token_revoked(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests token validation when the token has been revoked."""
    # Simulate a scenario where validate_token raises an error indicating revocation,
    # which falls into the generic exception handler.
    mock_auth_service.validate_token.side_effect = Exception("Token validation process failed unexpectedly (e.g., revocation check error)")

    mock_request = MagicMock()
    mock_request.token = "revoked.or.problematic.token"

    response = await auth_servicer.ValidateToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("revoked.or.problematic.token")
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_grpc_context.set_details.assert_called_once_with("Internal server error during token validation")

@pytest.mark.asyncio
async def test_validate_token_malformed_request(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests token validation when the request token is missing or malformed."""
    mock_auth_service.validate_token.side_effect = ValueError("Token must be provided")

    mock_request = MagicMock()
    mock_request.token = None # Simulate missing token

    response = await auth_servicer.ValidateToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with(None)
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_grpc_context.set_details.assert_called_once_with("Internal server error during token validation")

@pytest.mark.asyncio
async def test_refresh_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token refresh."""
    valid_refresh_token_data = create_mock_token_data(token_type="refresh", jti="refresh-jti")
    new_access_token_data = create_mock_token_data(token_type="access", jti="new-access-jti")

    mock_auth_service.validate_token.return_value = valid_refresh_token_data
    # Mock create_token_pair to return a new token pair
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "new.access.token"
    mock_token_pair.refresh_token = "new.refresh.token"
    mock_token_pair.token_type = "access"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair

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
    assert response.access_token == "new.access.token"
    assert response.token_type == "access"
    assert mock_grpc_context.set_code.assert_not_called()

@pytest.mark.asyncio
async def test_refresh_token_invalid_token_type(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests refreshing with a token that is not a refresh token."""
    access_token_data = create_mock_token_data(token_type="access")
    mock_auth_service.validate_token.return_value = access_token_data

    mock_request = MagicMock()
    mock_request.refresh_token = "not.a.refresh.token"

    response = await auth_servicer.RefreshToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("not.a.refresh.token")
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
    mock_grpc_context.set_details.assert_called_once_with("Token is not a refresh token")

@pytest.mark.asyncio
async def test_refresh_token_validation_failed(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests refresh token when token validation fails."""
    mock_auth_service.validate_token.side_effect = PyJWTError("Invalid token")

    mock_request = MagicMock()
    mock_request.refresh_token = "invalid.refresh.token"

    response = await auth_servicer.RefreshToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("invalid.refresh.token")
    assert response.valid is False
    mock_grpc_context.set_code.assert_not_called() # Error is handled internally by returning invalid response

@pytest.mark.asyncio
async def test_revoke_token_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful token revocation."""
    mock_auth_service.revoke_token.return_value = None

    mock_request = MagicMock()
    mock_request.token = "token.to.revoke"

    response = await auth_servicer.RevokeToken(mock_request, mock_grpc_context)

    mock_auth_service.revoke_token.assert_called_once_with("token.to.revoke")
    assert isinstance(response, empty_pb2.Empty) # Expecting an empty response message
    mock_grpc_context.set_code.assert_not_called()

@pytest.mark.asyncio
async def test_revoke_token_failure(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests token revocation failure."""
    mock_auth_service.revoke_token.side_effect = Exception("Failed to revoke token")

    mock_request = MagicMock()
    mock_request.token = "token.that.fails.revocation"

    response = await auth_servicer.RevokeToken(mock_request, mock_grpc_context)

    mock_auth_service.revoke_token.assert_called_once_with("token.that.fails.revocation")
    assert isinstance(response, empty_pb2.Empty) # Still returns empty, but error logged and code set
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

@pytest.mark.asyncio
async def test_get_user_info_success_from_db(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests GetUserInfo when data is fetched from the database."""
    mock_token_data = create_mock_token_data(user_id="db-user-id", tier="premium")
    mock_auth_service.validate_token.return_value = mock_token_data

    mock_db_cache.get_user.return_value = None # Ensure cache miss

    # Mock DB interaction
    mock_user = create_mock_user(id="db-user-id", email="db@example.com", tier="premium", full_name="DB User", mfa_enabled=True, last_login_at=datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC))
    
    # Mock the session and execute call
    mock_db_session = mock_db_manager # Use the fixture's mock session
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user

    # Mock timestamp creation as it's done inside the servicer
    with patch("src.auth.grpc_server.timestamp_pb2.Timestamp") as MockTimestamp, 
         patch("src.auth.grpc_server.MessageToDict") as MockMessageToDict:

        # Create mock timestamp objects
        ts_created = timestamp_pb2.Timestamp()
        ts_created.FromDatetime(mock_user.created_at)
        ts_last_login = timestamp_pb2.Timestamp()
        ts_last_login.FromDatetime(mock_user.last_login_at)
        
        MockTimestamp.side_effect = lambda: ts_created # Simplified for return value
        # Configure the side effect to return appropriate timestamps based on input
        def timestamp_side_effect(dt_obj=None):
            ts = timestamp_pb2.Timestamp()
            if dt_obj:
                ts.FromDatetime(dt_obj)
            return ts
        MockTimestamp.side_effect = timestamp_side_effect

        mock_request = MagicMock()
        mock_request.token = "valid.token.for.userinfo"

        response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

        mock_auth_service.validate_token.assert_called_once_with("valid.token.for.userinfo")
        mock_db_cache.get_user.assert_called_once_with("db-user-id")
        mock_db_session.execute.assert_called_once()
        mock_db_cache.set_user.assert_called_once() # Should be called to cache the user

        assert response.user_id == "db-user-id"
        assert response.email == "db@example.com"
        assert response.tier == "premium"
        assert response.full_name == "DB User"
        assert response.mfa_enabled is True
        assert response.created_at.seconds == int(mock_user.created_at.timestamp())
        assert response.last_login.seconds == int(mock_user.last_login_at.timestamp())
        assert response.roles == ["premium"] # Based on the assumption in servicer

@pytest.mark.asyncio
async def test_get_user_info_success_from_cache(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests GetUserInfo when data is fetched from the distributed cache."""
    mock_token_data = create_mock_token_data(user_id="cache-user-id", tier="enterprise")
    mock_auth_service.validate_token.return_value = mock_token_data

    # Mock cache hit
    mock_cached_user_dict = {
        "user_id": "cache-user-id",
        "email": "cache@example.com",
        "tier": "enterprise",
        "full_name": "Cache User",
        "mfa_enabled": True,
        "created_at": (datetime.now(UTC) - timedelta(days=10)).isoformat(),
        "last_login": (datetime.now(UTC) - timedelta(hours=5)).isoformat(),
        "roles": ["enterprise"]
    }
    mock_db_cache.get_user.return_value = mock_cached_user_dict

    # Create a mock UserInfo proto object that would be returned by ParseDict
    expected_user_info = auth_pb2.UserInfo(**{
        "user_id": "cache-user-id",
        "email": "cache@example.com",
        "tier": "enterprise",
        "full_name": "Cache User",
        "mfa_enabled": True,
        "created_at": timestamp_pb2.Timestamp(seconds=int((datetime.now(UTC) - timedelta(days=10)).timestamp())),
        "last_login": timestamp_pb2.Timestamp(seconds=int((datetime.now(UTC) - timedelta(hours=5)).timestamp())),
        "roles": ["enterprise"]
    })

    # Mock ParseDict to return the expected proto object when called
    with patch("src.auth.grpc_server.ParseDict", return_value=expected_user_info) as mock_parse_dict:
        mock_request = MagicMock()
        mock_request.token = "valid.token.for.cacheuser"

        response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

        mock_auth_service.validate_token.assert_called_once_with("valid.token.for.cacheuser")
        mock_db_cache.get_user.assert_called_once_with("cache-user-id")
        mock_db_manager.async_session_factory.assert_not_called() # DB should not be accessed
        mock_parse_dict.assert_called_once()
        assert response == expected_user_info

@pytest.mark.asyncio
async def test_get_user_info_user_not_found(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests GetUserInfo when the user is not found in DB."""
    mock_token_data = create_mock_token_data(user_id="nonexistent-user")
    mock_auth_service.validate_token.return_value = mock_token_data
    mock_db_cache.get_user.return_value = None

    mock_db_session = mock_db_manager
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None # User not found

    mock_request = MagicMock()
    mock_request.token = "token.for.nonexistent.user"

    response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("token.for.nonexistent.user")
    mock_db_cache.get_user.assert_called_once_with("nonexistent-user")
    mock_db_session.execute.assert_called_once()
    assert response.user_id == "" # Default empty response

@pytest.mark.asyncio
async def test_get_user_info_internal_error(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests GetUserInfo when an internal error occurs."""
    mock_auth_service.validate_token.side_effect = Exception("Unexpected error")

    mock_request = MagicMock()
    mock_request.token = "token.causing.error"

    response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("token.causing.error")
    mock_db_cache.get_user.assert_not_called()
    mock_db_manager.async_session_factory.assert_not_called()
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

@pytest.mark.asyncio
async def test_create_token_pair_success(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests successful creation of a token pair."""
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "new-access-token"
    mock_token_pair.refresh_token = "new-refresh-token"
    mock_token_pair.token_type = "Bearer"
    mock_token_pair.expires_in = 3600
    
    mock_auth_service.create_token_pair.return_value = mock_token_pair

    # Mock timestamp creation
    now = datetime.now(UTC)
    with patch("src.auth.grpc_server.timestamp_pb2.Timestamp") as MockTimestamp:
        mock_ts_instance = MagicMock()
        mock_ts_instance.seconds = int(now.timestamp())
        MockTimestamp.return_value = mock_ts_instance

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
        assert response.token_type == "Bearer"
        assert response.expires_in == 3600
        assert response.issued_at.seconds == int(now.timestamp())
        mock_grpc_context.set_code.assert_not_called()

@pytest.mark.asyncio
async def test_create_token_pair_failure(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests failure during token pair creation."""
    mock_auth_service.create_token_pair.side_effect = Exception("Token creation failed")

    mock_request = auth_pb2.CreateTokenRequest(
        user_id="user-fail-token",
        email="fail@example.com",
        tier="basic"
    )

    response = await auth_servicer.CreateTokenPair(mock_request, mock_grpc_context)

    mock_auth_service.create_token_pair.assert_called_once()
    assert response.access_token == "" # Default empty response
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

@pytest.mark.asyncio
async def test_validate_api_key_success(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests successful API key validation."""
    # Mock data for a valid API key
    mock_api_key_resp_proto = auth_pb2.APIKeyResponse(
        valid=True,
        user_id="api-user-1",
        email="api@example.com",
        tier="enterprise",
        key_name="my-api-key",
        created_at=timestamp_pb2.Timestamp(seconds=int(datetime.now(UTC).timestamp()))
    )
    
    # Mock cache hit for API key
    mock_cached_api_key_dict = MessageToDict(mock_api_key_resp_proto)
    mock_db_cache.get_api_key.return_value = mock_cached_api_key_dict

    mock_request = MagicMock()
    mock_request.api_key = "valid-api-key-secret"

    response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

    mock_db_cache.get_api_key.assert_called_once()
    mock_db_manager.async_session_factory.assert_not_called() # Cache hit, DB not accessed
    assert response.valid is True
    assert response.user_id == "api-user-1"
    assert response.tier == "enterprise"
    assert mock_grpc_context.set_code.assert_not_called()

@pytest.mark.asyncio
async def test_validate_api_key_success_from_db(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests API key validation when fetched from DB."""
    mock_auth_service.validate_token.return_value = create_mock_token_data(user_id="api-db-user-id", tier="premium") # Not used by ValidateAPIKey directly but for context.

    mock_db_cache.get_api_key.return_value = None # Cache miss

    # Mock DB interaction
    mock_user = create_mock_user(id="api-db-user-id", email="api-db@example.com", tier="premium")
    mock_api_key_record = MagicMock()
    mock_api_key_record.key_hash = "mock-hash"
    mock_api_key_record.name = "db-api-key-name"
    mock_api_key_record.created_at = datetime.now(UTC) - timedelta(days=5)
    mock_api_key_record.user = mock_user
    mock_api_key_record.is_active = True

    mock_db_session = mock_db_manager
    # Mock the query for APIKey and joinedload User
    mock_query = MagicMock()
    mock_query.options.return_value.where.return_value.scalar_one_or_none.return_value = mock_api_key_record
    mock_db_session.execute.return_value = mock_query

    # Mock timestamp creation
    created_at_ts = timestamp_pb2.Timestamp()
    created_at_ts.FromDatetime(mock_api_key_record.created_at)

    with patch("src.auth.grpc_server.timestamp_pb2.Timestamp", return_value=created_at_ts) as MockTimestamp:
        mock_request = MagicMock()
        mock_request.api_key = "secret-api-key-for-db" # This will be hashed

        response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

        mock_db_cache.get_api_key.assert_called_once()
        mock_db_session.execute.assert_called_once()
        mock_db_cache.set_api_key.assert_called_once() # Should be called to cache the result

        assert response.valid is True
        assert response.user_id == "api-db-user-id"
        assert response.email == "api-db@example.com"
        assert response.tier == "premium"
        assert response.key_name == "db-api-key-name"
        assert response.created_at.seconds == int(mock_api_key_record.created_at.timestamp())

@pytest.mark.asyncio
async def test_validate_api_key_invalid(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests API key validation when the key is invalid."""
    mock_db_cache.get_api_key.return_value = None
    mock_db_manager.async_session_factory.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = None # Simulate not found

    mock_request = MagicMock()
    mock_request.api_key = "invalid-api-key"

    response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

    mock_db_cache.get_api_key.assert_called_once()
    mock_db_manager.async_session_factory.assert_called_once()
    assert response.valid is False
    assert response.user_id == "" # Default empty response

@pytest.mark.asyncio
async def test_validate_api_key_inactive(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests API key validation when the key is inactive."""
    mock_db_cache.get_api_key.return_value = None
    
    mock_user = create_mock_user(id="inactive-user", email="inactive@example.com", tier="free")
    mock_api_key_record = MagicMock()
    mock_api_key_record.key_hash = "mock-hash"
    mock_api_key_record.name = "inactive-api-key-name"
    mock_api_key_record.created_at = datetime.now(UTC) - timedelta(days=5)
    mock_api_key_record.user = mock_user
    mock_api_key_record.is_active = False # Key is inactive

    mock_db_session = mock_db_manager
    mock_query = MagicMock()
    mock_query.options.return_value.where.return_value.scalar_one_or_none.return_value = mock_api_key_record
    mock_db_session.execute.return_value = mock_query

    mock_request = MagicMock()
    mock_request.api_key = "inactive-api-key"

    response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

    mock_db_cache.get_api_key.assert_called_once()
    mock_db_manager.async_session_factory.assert_called_once()
    assert response.valid is False
    assert response.user_id == ""

@pytest.mark.asyncio
async def test_introspect_token_success(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests successful token introspection."""
    mock_token_data = create_mock_token_data(
        user_id="intro-user-id",
        email="intro@example.com",
        tier="enterprise",
        token_type="bearer",
        scopes=["admin", "read"],
        jti="intro-jti"
    )
    mock_auth_service.validate_token.return_value = mock_token_data

    mock_request = MagicMock()
    mock_request.token = "valid.token.for.introspection"

    response = await auth_servicer.IntrospectToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.token.for.introspection")
    assert response.active is True
    assert response.sub == "intro-user-id"
    assert response.username == "intro@example.com"
    assert response.token_type == "bearer"
    assert response.scope == "admin read"
    assert response.iss == "manifold-auth-v2"
    assert response.iat == int(mock_token_data.iat.timestamp())
    assert response.exp == int(mock_token_data.exp.timestamp())

@pytest.mark.asyncio
async def test_introspect_token_invalid(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests token introspection when the token is invalid."""
    mock_auth_service.validate_token.side_effect = PyJWTError("Invalid token")

    mock_request = MagicMock()
    mock_request.token = "invalid.token.for.introspection"

    response = await auth_servicer.IntrospectToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("invalid.token.for.introspection")
    assert response.active is False

@pytest.mark.asyncio
async def test_introspect_token_expired(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests token introspection when the token is expired."""
    mock_auth_service.validate_token.side_effect = ExpiredSignatureError("Token expired")

    mock_request = MagicMock()
    mock_request.token = "expired.token.for.introspection"

    response = await auth_servicer.IntrospectToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("expired.token.for.introspection")
    assert response.active is False

# --- Test additions for GetUserInfo ---
@pytest.mark.asyncio
async def test_get_user_info_success_from_local_cache(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests GetUserInfo when data is fetched from the servicer's local cache."""
    mock_token_data = create_mock_token_data(user_id="local-cache-user-id", tier="free")
    mock_auth_service.validate_token.return_value = mock_token_data

    # Mock local cache hit
    mock_local_user_info = create_mock_user_info_proto(
        user_id="local-cache-user-id",
        email="local@example.com",
        tier="free",
        full_name="Local Cache User",
        mfa_enabled=False,
        created_at=datetime(2022, 5, 10, 12, 0, 0, tzinfo=UTC),
        last_login_at=datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
    )
    auth_servicer._user_cache[mock_token_data.user_id] = mock_local_user_info

    mock_request = MagicMock()
    mock_request.token = "valid.token.for.localcacheuser"

    response = await auth_servicer.GetUserInfo(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with("valid.token.for.localcacheuser")
    mock_db_cache.get_user.assert_not_called() # Local cache hit, so distributed cache and DB should not be called
    mock_db_manager.async_session_factory.assert_not_called()
    assert response == mock_local_user_info

@pytest.mark.asyncio
async def test_create_token_pair_missing_fields(auth_servicer, mock_auth_service, mock_grpc_context):
    """Tests CreateTokenPair when essential fields are missing (e.g., user_id)."""
    # In FastAPI/Pydantic, missing required fields would raise validation errors.
    # For gRPC, the request object might be incomplete. We'll simulate an error
    # occurring during processing, likely related to missing data.
    mock_auth_service.create_token_pair.side_effect = ValueError("Missing required fields for token creation")

    mock_request = auth_pb2.CreateTokenRequest(
        email="user@example.com", # Missing user_id
        tier="basic"
    )

    response = await auth_servicer.CreateTokenPair(mock_request, mock_grpc_context)

    mock_auth_service.create_token_pair.assert_called_once()
    assert response.access_token == ""
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL) # Assuming generic error handling for missing fields

@pytest.mark.asyncio
async def test_validate_api_key_network_error(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests API key validation when a network error occurs during DB access."""
    mock_db_cache.get_api_key.return_value = None # Cache miss

    # Simulate a network error when accessing the DB
    mock_db_manager.async_session_factory.side_effect = grpc.aio.InternalError("Network error connecting to DB")

    mock_request = MagicMock()
    mock_request.api_key = "api-key-network-error"

    response = await auth_servicer.ValidateAPIKey(mock_request, mock_grpc_context)

    mock_db_cache.get_api_key.assert_called_once()
    mock_db_manager.async_session_factory.assert_called_once()
    assert response.valid is False
    mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

@pytest.mark.asyncio
async def test_introspect_token_no_token(auth_servicer, mock_auth_service, mock_db_manager, mock_db_cache, mock_grpc_context):
    """Tests introspection when no token is provided."""
    # auth_service.validate_token should raise an error if token is None or empty
    mock_auth_service.validate_token.side_effect = ValueError("Token must be provided")

    mock_request = MagicMock()
    mock_request.token = None # No token provided

    response = await auth_servicer.IntrospectToken(mock_request, mock_grpc_context)

    mock_auth_service.validate_token.assert_called_once_with(None)
    assert response.active is False
