import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OptionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CALL: _ClassVar[OptionType]
    PUT: _ClassVar[OptionType]

class OrderSide(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BUY: _ClassVar[OrderSide]
    SELL: _ClassVar[OrderSide]

class AlertType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRICE_ALERT: _ClassVar[AlertType]
    TRADE_EXECUTION: _ClassVar[AlertType]
    RISK_WARNING: _ClassVar[AlertType]
    MARGIN_CALL: _ClassVar[AlertType]
    SYSTEM_NOTIFICATION: _ClassVar[AlertType]

class AlertSeverity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INFO: _ClassVar[AlertSeverity]
    WARNING: _ClassVar[AlertSeverity]
    ERROR: _ClassVar[AlertSeverity]
    CRITICAL: _ClassVar[AlertSeverity]
CALL: OptionType
PUT: OptionType
BUY: OrderSide
SELL: OrderSide
PRICE_ALERT: AlertType
TRADE_EXECUTION: AlertType
RISK_WARNING: AlertType
MARGIN_CALL: AlertType
SYSTEM_NOTIFICATION: AlertType
INFO: AlertSeverity
WARNING: AlertSeverity
ERROR: AlertSeverity
CRITICAL: AlertSeverity

class TickerUpdate(_message.Message):
    __slots__ = ("symbol", "price", "change", "volume", "timestamp")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    price: float
    change: float
    volume: int
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, symbol: _Optional[str] = ..., price: _Optional[float] = ..., change: _Optional[float] = ..., volume: _Optional[int] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class OptionChainSnapshot(_message.Message):
    __slots__ = ("underlying", "options", "snapshot_time")
    UNDERLYING_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_TIME_FIELD_NUMBER: _ClassVar[int]
    underlying: str
    options: _containers.RepeatedCompositeFieldContainer[OptionQuote]
    snapshot_time: _timestamp_pb2.Timestamp
    def __init__(self, underlying: _Optional[str] = ..., options: _Optional[_Iterable[_Union[OptionQuote, _Mapping]]] = ..., snapshot_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class OptionQuote(_message.Message):
    __slots__ = ("strike", "expiration", "type", "bid", "ask", "last", "volume", "open_interest", "implied_vol", "greeks")
    STRIKE_FIELD_NUMBER: _ClassVar[int]
    EXPIRATION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BID_FIELD_NUMBER: _ClassVar[int]
    ASK_FIELD_NUMBER: _ClassVar[int]
    LAST_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    OPEN_INTEREST_FIELD_NUMBER: _ClassVar[int]
    IMPLIED_VOL_FIELD_NUMBER: _ClassVar[int]
    GREEKS_FIELD_NUMBER: _ClassVar[int]
    strike: float
    expiration: str
    type: OptionType
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    greeks: Greeks
    def __init__(self, strike: _Optional[float] = ..., expiration: _Optional[str] = ..., type: _Optional[_Union[OptionType, str]] = ..., bid: _Optional[float] = ..., ask: _Optional[float] = ..., last: _Optional[float] = ..., volume: _Optional[int] = ..., open_interest: _Optional[int] = ..., implied_vol: _Optional[float] = ..., greeks: _Optional[_Union[Greeks, _Mapping]] = ...) -> None: ...

class Greeks(_message.Message):
    __slots__ = ("delta", "gamma", "theta", "vega", "rho")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    GAMMA_FIELD_NUMBER: _ClassVar[int]
    THETA_FIELD_NUMBER: _ClassVar[int]
    VEGA_FIELD_NUMBER: _ClassVar[int]
    RHO_FIELD_NUMBER: _ClassVar[int]
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    def __init__(self, delta: _Optional[float] = ..., gamma: _Optional[float] = ..., theta: _Optional[float] = ..., vega: _Optional[float] = ..., rho: _Optional[float] = ...) -> None: ...

class TradeExecution(_message.Message):
    __slots__ = ("order_id", "symbol", "price", "quantity", "side", "executed_at", "execution_id")
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    EXECUTED_AT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    order_id: str
    symbol: str
    price: float
    quantity: int
    side: OrderSide
    executed_at: _timestamp_pb2.Timestamp
    execution_id: str
    def __init__(self, order_id: _Optional[str] = ..., symbol: _Optional[str] = ..., price: _Optional[float] = ..., quantity: _Optional[int] = ..., side: _Optional[_Union[OrderSide, str]] = ..., executed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., execution_id: _Optional[str] = ...) -> None: ...

class PortfolioUpdate(_message.Message):
    __slots__ = ("portfolio_id", "total_value", "cash", "pnl_today", "pnl_total", "positions", "updated_at")
    PORTFOLIO_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    CASH_FIELD_NUMBER: _ClassVar[int]
    PNL_TODAY_FIELD_NUMBER: _ClassVar[int]
    PNL_TOTAL_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    portfolio_id: str
    total_value: float
    cash: float
    pnl_today: float
    pnl_total: float
    positions: _containers.RepeatedCompositeFieldContainer[Position]
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, portfolio_id: _Optional[str] = ..., total_value: _Optional[float] = ..., cash: _Optional[float] = ..., pnl_today: _Optional[float] = ..., pnl_total: _Optional[float] = ..., positions: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Position(_message.Message):
    __slots__ = ("symbol", "quantity", "avg_price", "current_price", "unrealized_pnl")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    AVG_PRICE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PRICE_FIELD_NUMBER: _ClassVar[int]
    UNREALIZED_PNL_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    def __init__(self, symbol: _Optional[str] = ..., quantity: _Optional[float] = ..., avg_price: _Optional[float] = ..., current_price: _Optional[float] = ..., unrealized_pnl: _Optional[float] = ...) -> None: ...

class Alert(_message.Message):
    __slots__ = ("alert_id", "type", "title", "message", "severity", "created_at", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ALERT_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    alert_id: str
    type: AlertType
    title: str
    message: str
    severity: AlertSeverity
    created_at: _timestamp_pb2.Timestamp
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, alert_id: _Optional[str] = ..., type: _Optional[_Union[AlertType, str]] = ..., title: _Optional[str] = ..., message: _Optional[str] = ..., severity: _Optional[_Union[AlertSeverity, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
