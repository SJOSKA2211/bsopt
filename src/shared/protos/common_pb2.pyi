import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "status", "metadata", "timestamp")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    metadata: _containers.ScalarMap[str, str]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, healthy: bool = ..., status: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class PaginationRequest(_message.Message):
    __slots__ = ("page", "page_size", "sort_by", "descending")
    PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_NUMBER: _ClassVar[int]
    DESCENDING_FIELD_NUMBER: _ClassVar[int]
    page: int
    page_size: int
    sort_by: str
    descending: bool
    def __init__(self, page: _Optional[int] = ..., page_size: _Optional[int] = ..., sort_by: _Optional[str] = ..., descending: bool = ...) -> None: ...

class PaginationResponse(_message.Message):
    __slots__ = ("total_items", "total_pages", "current_page", "page_size", "has_next", "has_previous")
    TOTAL_ITEMS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PAGES_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    HAS_NEXT_FIELD_NUMBER: _ClassVar[int]
    HAS_PREVIOUS_FIELD_NUMBER: _ClassVar[int]
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_previous: bool
    def __init__(self, total_items: _Optional[int] = ..., total_pages: _Optional[int] = ..., current_page: _Optional[int] = ..., page_size: _Optional[int] = ..., has_next: bool = ..., has_previous: bool = ...) -> None: ...

class TimestampRange(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: _timestamp_pb2.Timestamp
    end: _timestamp_pb2.Timestamp
    def __init__(self, start: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., end: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("code", "message", "details", "errors", "timestamp", "request_id")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    details: str
    errors: _containers.RepeatedCompositeFieldContainer[ErrorField]
    timestamp: _timestamp_pb2.Timestamp
    request_id: str
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., details: _Optional[str] = ..., errors: _Optional[_Iterable[_Union[ErrorField, _Mapping]]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class ErrorField(_message.Message):
    __slots__ = ("field", "message", "code")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    field: str
    message: str
    code: str
    def __init__(self, field: _Optional[str] = ..., message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ("tags", "counters", "gauges")
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class CountersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class GaugesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TAGS_FIELD_NUMBER: _ClassVar[int]
    COUNTERS_FIELD_NUMBER: _ClassVar[int]
    GAUGES_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.ScalarMap[str, str]
    counters: _containers.ScalarMap[str, int]
    gauges: _containers.ScalarMap[str, float]
    def __init__(self, tags: _Optional[_Mapping[str, str]] = ..., counters: _Optional[_Mapping[str, int]] = ..., gauges: _Optional[_Mapping[str, float]] = ...) -> None: ...

class BatchRequest(_message.Message):
    __slots__ = ("ids", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    IDS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class BatchResponse(_message.Message):
    __slots__ = ("results", "errors", "success_count", "failure_count")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Value]
    errors: _containers.RepeatedCompositeFieldContainer[ErrorResponse]
    success_count: int
    failure_count: int
    def __init__(self, results: _Optional[_Iterable[_Union[_struct_pb2.Value, _Mapping]]] = ..., errors: _Optional[_Iterable[_Union[ErrorResponse, _Mapping]]] = ..., success_count: _Optional[int] = ..., failure_count: _Optional[int] = ...) -> None: ...

class VersionInfo(_message.Message):
    __slots__ = ("version", "build_hash", "build_date", "go_version", "os", "arch")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    BUILD_HASH_FIELD_NUMBER: _ClassVar[int]
    BUILD_DATE_FIELD_NUMBER: _ClassVar[int]
    GO_VERSION_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    ARCH_FIELD_NUMBER: _ClassVar[int]
    version: str
    build_hash: str
    build_date: str
    go_version: str
    os: str
    arch: str
    def __init__(self, version: _Optional[str] = ..., build_hash: _Optional[str] = ..., build_date: _Optional[str] = ..., go_version: _Optional[str] = ..., os: _Optional[str] = ..., arch: _Optional[str] = ...) -> None: ...

class RateLimitInfo(_message.Message):
    __slots__ = ("limit", "remaining", "reset_time", "tier")
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    REMAINING_FIELD_NUMBER: _ClassVar[int]
    RESET_TIME_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    limit: int
    remaining: int
    reset_time: int
    tier: str
    def __init__(self, limit: _Optional[int] = ..., remaining: _Optional[int] = ..., reset_time: _Optional[int] = ..., tier: _Optional[str] = ...) -> None: ...

class QuotaInfo(_message.Message):
    __slots__ = ("resource", "used", "limit", "utilization_percent")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    USED_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    resource: str
    used: int
    limit: int
    utilization_percent: float
    def __init__(self, resource: _Optional[str] = ..., used: _Optional[int] = ..., limit: _Optional[int] = ..., utilization_percent: _Optional[float] = ...) -> None: ...
