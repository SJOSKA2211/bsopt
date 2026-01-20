"""
SQLAlchemy ORM Models for BSOPT Platform

This module defines all database models with:
- Relationship mappings with eager loading
- Indexes matching schema.sql
- Constraints for data integrity
- Type hints for IDE support
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# =============================================================================
# USER MODEL
# =============================================================================


class User(Base):
    """User accounts with tiered access (free, pro, enterprise)."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(String(20), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # CRITICAL: MFA secrets should NEVER be stored in plaintext.
    # This field needs strong encryption at rest, or ideally, a secure key management system/HSM.
    # A derived key should be stored, not the secret itself.
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(255))
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_token: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    # Relationships
    portfolios: Mapped[List["Portfolio"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    orders: Mapped[List["Order"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="dynamic"
    )
    ml_models: Mapped[List["MLModel"]] = relationship(
        back_populates="created_by_user", cascade="all, delete-orphan", lazy="selectin"
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


# =============================================================================
# API KEY MODEL
# =============================================================================


class APIKey(Base):
    """Secure API keys for programmatic access."""

    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_keys_user_id", "user_id"),
        Index("idx_api_keys_key_hash", "key_hash"),
    )

    def __repr__(self) -> str:
        return f"<APIKey(name={self.name}, prefix={self.prefix}...)>"


# =============================================================================
# OPTIONS PRICES MODEL (TimescaleDB Hypertable)
# =============================================================================


class OptionPrice(Base):
    """Time-series options market data (TimescaleDB hypertable)."""

    __tablename__ = "options_prices"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    strike: Mapped[Decimal] = mapped_column(Numeric(12, 2), primary_key=True, nullable=False)
    expiry: Mapped[date] = mapped_column(Date, primary_key=True, nullable=False)
    option_type: Mapped[str] = mapped_column(String(4), primary_key=True, nullable=False)

    bid: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ask: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    last: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    volume: Mapped[Optional[int]] = mapped_column(Integer)
    open_interest: Mapped[Optional[int]] = mapped_column(Integer)

    implied_volatility: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    delta: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    gamma: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    vega: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    theta: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    rho: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))

    __table_args__ = (
        CheckConstraint("option_type IN ('call', 'put')", name="check_option_type"),
        Index("idx_options_prices_symbol_time", "symbol", "time"),
        Index("idx_options_prices_expiry_time", "expiry", "time"),
        Index("idx_options_prices_chain", "time", "symbol", "expiry", "option_type", "strike"),
    )

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    def __repr__(self) -> str:
        return f"<OptionPrice({self.symbol} {self.strike} {self.option_type} @ {self.time})>"


# =============================================================================
# MARKET TICKS MODEL (TimescaleDB Hypertable)
# =============================================================================


class MarketTick(Base):
    """Raw underlying market ticks (TimescaleDB hypertable) for TFT group_ids."""

    __tablename__ = "market_ticks"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_market_ticks_symbol_time", "symbol", "time"),
        Index("idx_market_ticks_time", "time"),
    )

    def __repr__(self) -> str:
        return f"<MarketTick({self.symbol} @ {self.time}: {self.price})>"


# =============================================================================
# PORTFOLIO MODEL
# =============================================================================


class Portfolio(Base):
    """User portfolios for position tracking."""

    __tablename__ = "portfolios"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=Decimal("0.00"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="portfolios")
    positions: Mapped[List["Position"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    orders: Mapped[List["Order"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="unique_user_portfolio"),
        CheckConstraint("cash_balance >= 0", name="check_positive_cash"),
        Index("idx_portfolios_user_created", "user_id", "created_at"),
        Index("idx_portfolios_user_name", "user_id", "name"),
    )

    @property
    def open_positions(self) -> List["Position"]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def total_value(self) -> Decimal:
        position_value = sum(p.quantity * p.entry_price for p in self.open_positions)
        return self.cash_balance + position_value

    def __repr__(self) -> str:
        return f"<Portfolio(id={self.id}, name={self.name}, cash={self.cash_balance})>"


# =============================================================================
# POSITION MODEL
# =============================================================================


class Position(Base):
    """Individual option positions (quantity > 0 = long, < 0 = short)."""

    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    strike: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    expiry: Mapped[Optional[date]] = mapped_column(Date)
    option_type: Mapped[Optional[str]] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    exit_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    status: Mapped[str] = mapped_column(String(10), default="open")

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")

    __table_args__ = (
        CheckConstraint("quantity != 0", name="check_nonzero_quantity"),
        CheckConstraint("status IN ('open', 'closed')", name="check_position_status"),
        CheckConstraint(
            "option_type IS NULL OR option_type IN ('call', 'put')",
            name="check_position_option_type",
        ),
        CheckConstraint(
            "(exit_price IS NULL AND exit_date IS NULL) OR "
            "(exit_price IS NOT NULL AND exit_date IS NOT NULL)",
            name="exit_price_requires_exit_date",
        ),
        Index("idx_positions_portfolio_status", "portfolio_id", "status"),
        Index("idx_positions_symbol_status", "symbol", "status"),
        Index("idx_positions_expiry_status", "expiry", "status"),
    )

    @property
    def is_option(self) -> bool:
        return self.option_type is not None

    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        return None

    def close(self, exit_price: Decimal, exit_date: Optional[datetime] = None) -> None:
        self.exit_price = exit_price
        self.exit_date = exit_date or datetime.now(timezone.utc)
        self.status = "closed"
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity

    def __repr__(self) -> str:
        return f"<Position({self.symbol} qty={self.quantity} status={self.status})>"


# =============================================================================
# ORDER MODEL
# =============================================================================


class Order(Base):
    """Trading order management (market, limit, stop, stop_limit)."""

    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    portfolio_id: Mapped[UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    strike: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    expiry: Mapped[Optional[date]] = mapped_column(Date)
    option_type: Mapped[Optional[str]] = mapped_column(String(4))
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    order_type: Mapped[str] = mapped_column(String(15), nullable=False)
    limit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    stop_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    broker: Mapped[Optional[str]] = mapped_column(String(50))
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="orders")
    portfolio: Mapped["Portfolio"] = relationship(back_populates="orders")

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="check_order_side"),
        CheckConstraint("quantity > 0", name="check_positive_quantity"),
        CheckConstraint(
            "order_type IN ('market', 'limit', 'stop', 'stop_limit')", name="check_order_type"
        ),
        CheckConstraint(
            "status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected')",
            name="check_order_status",
        ),
        CheckConstraint(
            "option_type IS NULL OR option_type IN ('call', 'put')", name="check_order_option_type"
        ),
        CheckConstraint(
            "(order_type != 'limit' AND order_type != 'stop_limit') OR limit_price IS NOT NULL",
            name="limit_order_requires_limit_price",
        ),
        CheckConstraint(
            "(order_type != 'stop' AND order_type != 'stop_limit') OR stop_price IS NOT NULL",
            name="stop_order_requires_stop_price",
        ),
        Index("idx_orders_user_created", "user_id", "created_at"),
        Index("idx_orders_portfolio_created", "portfolio_id", "created_at"),
        Index("idx_orders_status_created", "status", "created_at"),
        Index("idx_orders_broker_lookup", "broker", "broker_order_id"),
        Index("idx_orders_symbol_status", "symbol", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Order({self.side} {self.quantity} {self.symbol} @ {self.order_type})>"


# =============================================================================
# ML MODEL
# =============================================================================


class MLModel(Base):
    """ML model registry with versioning."""

    __tablename__ = "ml_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)
    training_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    model_artifact_url: Mapped[Optional[str]] = mapped_column(String(500))
    created_by: Mapped[Optional[UUID]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    created_by_user: Mapped[Optional["User"]] = relationship(back_populates="ml_models")
    predictions: Mapped[List["ModelPrediction"]] = relationship(
        back_populates="model", lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint("name", "version", name="unique_model_version"),
        CheckConstraint(
            "algorithm IN ("
            "'xgboost', 'lightgbm', 'neural_network', "
            "'random_forest', 'svm', 'ensemble'"
            ")",
            name="check_algorithm",
        ),
        CheckConstraint("version > 0", name="check_positive_version"),
        Index("idx_ml_models_production", "name", "is_production"),
        Index("idx_ml_models_version", "name", "version"),
        Index("idx_ml_models_created_by", "created_by"),
    )

    def __repr__(self) -> str:
        return f"<MLModel({self.name} v{self.version} [{self.algorithm}])>"


# =============================================================================
# MODEL PREDICTION
# =============================================================================


class ModelPrediction(Base):
    """Prediction logs for ML model monitoring."""

    __tablename__ = "model_predictions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("ml_models.id", ondelete="SET NULL")
    )
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    actual_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    prediction_error: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    model: Mapped[Optional["MLModel"]] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("idx_model_predictions_model_time", "model_id", "timestamp"),
        Index(
            "idx_model_predictions_pending", "timestamp", postgresql_where="actual_price IS NULL"
        ),
    )

    def calculate_error(self) -> Optional[Decimal]:
        if self.actual_price:
            self.prediction_error = abs(self.predicted_price - self.actual_price)
            return self.prediction_error
        return None

    def __repr__(self) -> str:
        return f"<ModelPrediction(predicted={self.predicted_price}, actual={self.actual_price})>"


# =============================================================================
# RATE LIMIT
# =============================================================================


class RateLimit(Base):
    """API rate limiting tracking by user and endpoint."""

    __tablename__ = "rate_limits"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    endpoint: Mapped[str] = mapped_column(String(100), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (Index("idx_rate_limits_lookup", "user_id", "endpoint", "window_start"),)

    def __repr__(self) -> str:
        return (
            f"<RateLimit(user={self.user_id}, "
            f"endpoint={self.endpoint}, count={self.request_count})>"
        )


# =============================================================================
# AUDIT LOG
# =============================================================================


class AuditLog(Base):
    """Security audit trail for compliance and monitoring."""

    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    user_email: Mapped[Optional[str]] = mapped_column(String(255))
    source_ip: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))
    request_path: Mapped[Optional[str]] = mapped_column(String(500))
    request_method: Mapped[Optional[str]] = mapped_column(String(10))
    # SECURITY: If user-controlled data is inserted into this JSON field without
    # proper sanitization, it could lead to JSON injection vulnerabilities.
    # Ensure strict validation and escaping of user input before embedding in JSON.
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_audit_logs_event_type", "event_type"),
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_created_at", "created_at"),
        Index("idx_audit_logs_source_ip", "source_ip"),
        Index("idx_audit_logs_event_user_time", "event_type", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog({self.event_type} user={self.user_id} @ {self.created_at})>"


# =============================================================================
# REQUEST LOG
# =============================================================================


class RequestLog(Base):
    """HTTP request/response logging for observability."""

    __tablename__ = "request_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    # SECURITY: query_params can contain user-controlled input.
    # Ensure proper output encoding (e.g., HTML entity encoding) when displayed
    # in web interfaces to prevent XSS. Sanitize before displaying in logs to prevent log injection.
    query_params: Mapped[Optional[str]] = mapped_column(Text)
    # SECURITY: If user-controlled header values are inserted into this JSON field
    # without proper sanitization, it could lead to JSON injection vulnerabilities.
    # Ensure strict validation and escaping of user input before embedding in JSON.
    headers: Mapped[Optional[dict]] = mapped_column(JSON)
    client_ip: Mapped[Optional[str]] = mapped_column(String(45))
    user_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    response_time_ms: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3))
    # SECURITY: If user-controlled input can be reflected into error messages,
    # it could lead to log injection. Ensure proper sanitization or use structured
    # logging with keyword arguments to prevent this.
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_request_logs_request_id", "request_id"),
        Index("idx_request_logs_path", "path"),
        Index("idx_request_logs_user_id", "user_id"),
        Index("idx_request_logs_status_code", "status_code"),
        Index("idx_request_logs_created_at", "created_at"),
        Index(
            "idx_request_logs_slow_requests",
            "response_time_ms",
            postgresql_where="response_time_ms > 1000",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<RequestLog({self.method} {self.path} -> "
            f"{self.status_code} in {self.response_time_ms}ms)>"
        )


# =============================================================================
# COMPLIANCE AND SECURITY MODELS
# =============================================================================


class SecurityIncident(Base):
    """Log of security incidents for breach notification and analysis."""

    __tablename__ = "security_incidents"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    reported_to_dpa: Mapped[bool] = mapped_column(Boolean, default=False)
    reported_to_dpa_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    nature_of_breach: Mapped[str] = mapped_column(Text, nullable=False)
    data_categories_affected: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    approximate_number_data_subjects: Mapped[int] = mapped_column(Integer, nullable=False)
    approximate_number_records: Mapped[int] = mapped_column(Integer, nullable=False)
    likely_consequences: Mapped[str] = mapped_column(Text, nullable=False)
    measures_taken: Mapped[str] = mapped_column(Text, nullable=False)

    data_subjects_notified: Mapped[bool] = mapped_column(Boolean, default=False)
    data_subjects_notified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    notification_method: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (
        Index("idx_security_incidents_severity", "severity"),
        Index("idx_security_incidents_type", "incident_type"),
        Index("idx_security_incidents_detected_at", "detected_at"),
    )

    def __repr__(self) -> str:
        return f"<SecurityIncident({self.incident_type} at {self.detected_at})>"


class GDPRRequest(Base):
    """Audit trail for GDPR data subject requests."""

    __tablename__ = "gdpr_requests"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), default="pending")
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    response_delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_gdpr_requests_user_status", "user_id", "status"),
        Index("idx_gdpr_requests_type_status", "request_type", "status"),
        Index("idx_gdpr_requests_deadline", "deadline"),
    )

    def __repr__(self) -> str:
        return f"<GDPRRequest(user={self.user_id}, type={self.request_type}, status={self.status})>"


# =============================================================================
# CALIBRATION RESULT
# =============================================================================


class CalibrationResult(Base):
    """Historical Heston and SVI calibration results."""

    __tablename__ = "calibration_results"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Heston Parameters
    v0: Mapped[float] = mapped_column(Numeric(12, 6))
    kappa: Mapped[float] = mapped_column(Numeric(12, 6))
    theta: Mapped[float] = mapped_column(Numeric(12, 6))
    sigma: Mapped[float] = mapped_column(Numeric(12, 6))
    rho: Mapped[float] = mapped_column(Numeric(12, 6))

    # Metrics
    rmse: Mapped[float] = mapped_column(Numeric(12, 6))
    r_squared: Mapped[float] = mapped_column(Numeric(12, 6))
    num_options: Mapped[int] = mapped_column(Integer)

    # SVI Parameters (JSON for surface)
    svi_params: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index("idx_calibration_symbol_time", "symbol", "time"),
    )

    def __repr__(self) -> str:
        return f"<CalibrationResult({self.symbol} @ {self.time}: RMSE={self.rmse})>"