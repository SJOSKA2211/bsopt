import datetime
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class TokenRequest(_message.Message):
    __slots__ = ("token",)
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    token: str
    def __init__(self, token: str | None = ...) -> None: ...

class TokenResponse(_message.Message):
    __slots__ = ("email", "expires_at", "issued_at", "roles", "tier", "token_type", "user_id", "valid")
    VALID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    user_id: str
    email: str
    tier: str
    expires_at: int
    issued_at: int
    token_type: str
    roles: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, valid: bool = ..., user_id: str | None = ..., email: str | None = ..., tier: str | None = ..., expires_at: int | None = ..., issued_at: int | None = ..., token_type: str | None = ..., roles: _Iterable[str] | None = ...) -> None: ...

class RefreshRequest(_message.Message):
    __slots__ = ("refresh_token",)
    REFRESH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    refresh_token: str
    def __init__(self, refresh_token: str | None = ...) -> None: ...

class RevokeRequest(_message.Message):
    __slots__ = ("token", "token_type_hint")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_HINT_FIELD_NUMBER: _ClassVar[int]
    token: str
    token_type_hint: str
    def __init__(self, token: str | None = ..., token_type_hint: str | None = ...) -> None: ...

class UserInfo(_message.Message):
    __slots__ = ("created_at", "email", "full_name", "is_verified", "last_login", "metadata", "mfa_enabled", "roles", "tier", "user_id")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_VERIFIED_FIELD_NUMBER: _ClassVar[int]
    MFA_ENABLED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_LOGIN_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    email: str
    tier: str
    full_name: str
    is_verified: bool
    mfa_enabled: bool
    created_at: _timestamp_pb2.Timestamp
    last_login: _timestamp_pb2.Timestamp
    roles: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, user_id: str | None = ..., email: str | None = ..., tier: str | None = ..., full_name: str | None = ..., is_verified: bool = ..., mfa_enabled: bool = ..., created_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., last_login: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., roles: _Iterable[str] | None = ..., metadata: _Mapping[str, str] | None = ...) -> None: ...

class CreateTokenRequest(_message.Message):
    __slots__ = ("access_token_minutes", "email", "refresh_token_days", "scopes", "tier", "user_id")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    ACCESS_TOKEN_MINUTES_FIELD_NUMBER: _ClassVar[int]
    REFRESH_TOKEN_DAYS_FIELD_NUMBER: _ClassVar[int]
    SCOPES_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    email: str
    tier: str
    access_token_minutes: int
    refresh_token_days: int
    scopes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, user_id: str | None = ..., email: str | None = ..., tier: str | None = ..., access_token_minutes: int | None = ..., refresh_token_days: int | None = ..., scopes: _Iterable[str] | None = ...) -> None: ...

class TokenPairResponse(_message.Message):
    __slots__ = ("access_token", "expires_in", "issued_at", "refresh_token", "token_type")
    ACCESS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    REFRESH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_IN_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    issued_at: _timestamp_pb2.Timestamp
    def __init__(self, access_token: str | None = ..., refresh_token: str | None = ..., token_type: str | None = ..., expires_in: int | None = ..., issued_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class APIKeyRequest(_message.Message):
    __slots__ = ("api_key",)
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    def __init__(self, api_key: str | None = ...) -> None: ...

class APIKeyResponse(_message.Message):
    __slots__ = ("created_at", "email", "expires_at", "key_name", "scopes", "tier", "user_id", "valid")
    VALID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    KEY_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    SCOPES_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    user_id: str
    email: str
    tier: str
    key_name: str
    created_at: _timestamp_pb2.Timestamp
    expires_at: _timestamp_pb2.Timestamp
    scopes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, valid: bool = ..., user_id: str | None = ..., email: str | None = ..., tier: str | None = ..., key_name: str | None = ..., created_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., expires_at: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., scopes: _Iterable[str] | None = ...) -> None: ...

class IntrospectionResponse(_message.Message):
    __slots__ = ("active", "client_id", "exp", "iat", "iss", "scope", "sub", "token_type", "username")
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    SUB_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_FIELD_NUMBER: _ClassVar[int]
    EXP_FIELD_NUMBER: _ClassVar[int]
    IAT_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    ISS_FIELD_NUMBER: _ClassVar[int]
    active: bool
    sub: str
    client_id: str
    username: str
    token_type: str
    exp: int
    iat: int
    scope: str
    iss: str
    def __init__(self, active: bool = ..., sub: str | None = ..., client_id: str | None = ..., username: str | None = ..., token_type: str | None = ..., exp: int | None = ..., iat: int | None = ..., scope: str | None = ..., iss: str | None = ...) -> None: ...

class AuthChallenge(_message.Message):
    __slots__ = ("challenge", "method", "session_id")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    method: str
    challenge: str
    session_id: str
    def __init__(self, method: str | None = ..., challenge: str | None = ..., session_id: str | None = ...) -> None: ...

class AuthResponse(_message.Message):
    __slots__ = ("authenticated", "factors_verified", "user_id")
    AUTHENTICATED_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    FACTORS_VERIFIED_FIELD_NUMBER: _ClassVar[int]
    authenticated: bool
    user_id: str
    factors_verified: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, authenticated: bool = ..., user_id: str | None = ..., factors_verified: _Iterable[str] | None = ...) -> None: ...
