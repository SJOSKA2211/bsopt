import datetime
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: str | None = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "metadata", "status", "timestamp")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    metadata: _containers.ScalarMap[str, str]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, healthy: bool = ..., status: str | None = ..., metadata: _Mapping[str, str] | None = ..., timestamp: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class PaginationRequest(_message.Message):
    __slots__ = ("descending", "page", "page_size", "sort_by")
    PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_NUMBER: _ClassVar[int]
    DESCENDING_FIELD_NUMBER: _ClassVar[int]
    page: int
    page_size: int
    sort_by: str
    descending: bool
    def __init__(self, page: int | None = ..., page_size: int | None = ..., sort_by: str | None = ..., descending: bool = ...) -> None: ...

class PaginationResponse(_message.Message):
    __slots__ = ("current_page", "has_next", "has_previous", "page_size", "total_items", "total_pages")
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
    def __init__(self, total_items: int | None = ..., total_pages: int | None = ..., current_page: int | None = ..., page_size: int | None = ..., has_next: bool = ..., has_previous: bool = ...) -> None: ...

class TimestampRange(_message.Message):
    __slots__ = ("end", "start")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: _timestamp_pb2.Timestamp
    end: _timestamp_pb2.Timestamp
    def __init__(self, start: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., end: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ...) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("code", "details", "errors", "message", "request_id", "timestamp")
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
    def __init__(self, code: str | None = ..., message: str | None = ..., details: str | None = ..., errors: _Iterable[ErrorField | _Mapping] | None = ..., timestamp: datetime.datetime | _timestamp_pb2.Timestamp | _Mapping | None = ..., request_id: str | None = ...) -> None: ...

class ErrorField(_message.Message):
    __slots__ = ("code", "field", "message")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    field: str
    message: str
    code: str
    def __init__(self, field: str | None = ..., message: str | None = ..., code: str | None = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ("counters", "gauges", "tags")
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    class CountersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...
    class GaugesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: str | None = ..., value: float | None = ...) -> None: ...
    TAGS_FIELD_NUMBER: _ClassVar[int]
    COUNTERS_FIELD_NUMBER: _ClassVar[int]
    GAUGES_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.ScalarMap[str, str]
    counters: _containers.ScalarMap[str, int]
    gauges: _containers.ScalarMap[str, float]
    def __init__(self, tags: _Mapping[str, str] | None = ..., counters: _Mapping[str, int] | None = ..., gauges: _Mapping[str, float] | None = ...) -> None: ...

class BatchRequest(_message.Message):
    __slots__ = ("ids", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    IDS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, ids: _Iterable[str] | None = ..., parameters: _Mapping[str, str] | None = ...) -> None: ...

class BatchResponse(_message.Message):
    __slots__ = ("errors", "failure_count", "results", "success_count")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Value]
    errors: _containers.RepeatedCompositeFieldContainer[ErrorResponse]
    success_count: int
    failure_count: int
    def __init__(self, results: _Iterable[_struct_pb2.Value | _Mapping] | None = ..., errors: _Iterable[ErrorResponse | _Mapping] | None = ..., success_count: int | None = ..., failure_count: int | None = ...) -> None: ...

class VersionInfo(_message.Message):
    __slots__ = ("arch", "build_date", "build_hash", "go_version", "os", "version")
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
    def __init__(self, version: str | None = ..., build_hash: str | None = ..., build_date: str | None = ..., go_version: str | None = ..., os: str | None = ..., arch: str | None = ...) -> None: ...

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
    def __init__(self, limit: int | None = ..., remaining: int | None = ..., reset_time: int | None = ..., tier: str | None = ...) -> None: ...

class QuotaInfo(_message.Message):
    __slots__ = ("limit", "resource", "used", "utilization_percent")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    USED_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    resource: str
    used: int
    limit: int
    utilization_percent: float
    def __init__(self, resource: str | None = ..., used: int | None = ..., limit: int | None = ..., utilization_percent: float | None = ...) -> None: ...
