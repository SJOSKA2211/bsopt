import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TokenRequest(_message.Message):
    __slots__ = ("token",)
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    token: str
    def __init__(self, token: _Optional[str] = ...) -> None: ...

class TokenResponse(_message.Message):
    __slots__ = ("valid", "user_id", "email", "tier", "expires_at", "issued_at", "token_type", "roles")
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
    def __init__(self, valid: bool = ..., user_id: _Optional[str] = ..., email: _Optional[str] = ..., tier: _Optional[str] = ..., expires_at: _Optional[int] = ..., issued_at: _Optional[int] = ..., token_type: _Optional[str] = ..., roles: _Optional[_Iterable[str]] = ...) -> None: ...

class RefreshRequest(_message.Message):
    __slots__ = ("refresh_token",)
    REFRESH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    refresh_token: str
    def __init__(self, refresh_token: _Optional[str] = ...) -> None: ...

class RevokeRequest(_message.Message):
    __slots__ = ("token", "token_type_hint")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_HINT_FIELD_NUMBER: _ClassVar[int]
    token: str
    token_type_hint: str
    def __init__(self, token: _Optional[str] = ..., token_type_hint: _Optional[str] = ...) -> None: ...

class UserInfo(_message.Message):
    __slots__ = ("user_id", "email", "tier", "full_name", "is_verified", "mfa_enabled", "created_at", "last_login", "roles", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
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
    def __init__(self, user_id: _Optional[str] = ..., email: _Optional[str] = ..., tier: _Optional[str] = ..., full_name: _Optional[str] = ..., is_verified: bool = ..., mfa_enabled: bool = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_login: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., roles: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class CreateTokenRequest(_message.Message):
    __slots__ = ("user_id", "email", "tier", "access_token_minutes", "refresh_token_days", "scopes")
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
    def __init__(self, user_id: _Optional[str] = ..., email: _Optional[str] = ..., tier: _Optional[str] = ..., access_token_minutes: _Optional[int] = ..., refresh_token_days: _Optional[int] = ..., scopes: _Optional[_Iterable[str]] = ...) -> None: ...

class TokenPairResponse(_message.Message):
    __slots__ = ("access_token", "refresh_token", "token_type", "expires_in", "issued_at")
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
    def __init__(self, access_token: _Optional[str] = ..., refresh_token: _Optional[str] = ..., token_type: _Optional[str] = ..., expires_in: _Optional[int] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class APIKeyRequest(_message.Message):
    __slots__ = ("api_key",)
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    def __init__(self, api_key: _Optional[str] = ...) -> None: ...

class APIKeyResponse(_message.Message):
    __slots__ = ("valid", "user_id", "email", "tier", "key_name", "created_at", "expires_at", "scopes")
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
    def __init__(self, valid: bool = ..., user_id: _Optional[str] = ..., email: _Optional[str] = ..., tier: _Optional[str] = ..., key_name: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., scopes: _Optional[_Iterable[str]] = ...) -> None: ...

class IntrospectionResponse(_message.Message):
    __slots__ = ("active", "sub", "client_id", "username", "token_type", "exp", "iat", "scope", "iss")
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
    def __init__(self, active: bool = ..., sub: _Optional[str] = ..., client_id: _Optional[str] = ..., username: _Optional[str] = ..., token_type: _Optional[str] = ..., exp: _Optional[int] = ..., iat: _Optional[int] = ..., scope: _Optional[str] = ..., iss: _Optional[str] = ...) -> None: ...

class AuthChallenge(_message.Message):
    __slots__ = ("method", "challenge", "session_id")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    method: str
    challenge: str
    session_id: str
    def __init__(self, method: _Optional[str] = ..., challenge: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class AuthResponse(_message.Message):
    __slots__ = ("authenticated", "user_id", "factors_verified")
    AUTHENTICATED_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    FACTORS_VERIFIED_FIELD_NUMBER: _ClassVar[int]
    authenticated: bool
    user_id: str
    factors_verified: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, authenticated: bool = ..., user_id: _Optional[str] = ..., factors_verified: _Optional[_Iterable[str]] = ...) -> None: ...
