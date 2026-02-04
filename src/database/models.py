"""
SQLAlchemy ORM Models for BSOPT Platform (Neon Optimized)
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
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# =============================================================================
# USER MODEL
# =============================================================================


class User(Base):
    """User accounts with tiered access."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(String(20), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    portfolios: Mapped[List["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    oauth_clients: Mapped[List["OAuth2Client"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    api_keys: Mapped[List["APIKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


# =============================================================================
# API KEY MODEL
# =============================================================================


class APIKey(Base):
    """Secure API keys for automated access."""

    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<APIKey(name={self.name}, prefix={self.key_prefix})>"


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================


class AuditLog(Base):
    """Audit logs for security and compliance."""

    __tablename__ = "audit_logs"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, server_default=func.now())
    method: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_ip: Mapped[str] = mapped_column(Text, nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)

    def __repr__(self) -> str:
        return f"<AuditLog(user_id={self.user_id}, path={self.path})>"


class RequestLog(Base):
    """Detailed API request logs."""

    __tablename__ = "request_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric, nullable=False)


class SecurityIncident(Base):
    """Security incident tracking."""

    __tablename__ = "security_incidents"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


# =============================================================================
# OAUTH2 CLIENT MODEL
# =============================================================================


class OAuth2Client(Base):
    """OAuth2 Client Registry."""

    __tablename__ = "oauth2_clients"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    client_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    redirect_uris: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    scopes: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    user: Mapped["User"] = relationship(back_populates="oauth_clients")

    def verify_secret(self, secret: str) -> bool:
        # Optimization: Use a real hash check here in production
        return self.client_secret == secret

    def __repr__(self) -> str:
        return f"<OAuth2Client(id={self.client_id})>"


# =============================================================================
# OPTIONS PRICES MODEL (Partitioned)
# =============================================================================


class OptionPrice(Base):
    """Time-series options market data."""

    __tablename__ = "options_prices"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    strike: Mapped[Decimal] = mapped_column(Numeric(12, 2), primary_key=True)
    expiry: Mapped[date] = mapped_column(Date, primary_key=True)
    option_type: Mapped[str] = mapped_column(String(4), primary_key=True)

    bid: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ask: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    last: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    volume: Mapped[Optional[int]] = mapped_column(Integer)
    open_interest: Mapped[Optional[int]] = mapped_column(Integer)
    implied_volatility: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))

    def __repr__(self) -> str:
        return f"<OptionPrice({self.symbol} @ {self.time})>"


# =============================================================================
# PORTFOLIO MODEL
# =============================================================================


class Portfolio(Base):
    """User portfolios."""

    __tablename__ = "portfolios"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=Decimal("0.00"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="portfolios")

    __table_args__ = (UniqueConstraint("user_id", "name"),)

    def __repr__(self) -> str:
        return f"<Portfolio(name={self.name})>"


# =============================================================================
# TRADING MODELS
# =============================================================================


class Position(Base):
    """Trading positions."""

    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(15, 4), default=Decimal("0"))
    average_price: Mapped[Decimal] = mapped_column(Numeric(15, 4), default=Decimal("0"))

    def __repr__(self) -> str:
        return f"<Position(symbol={self.symbol}, quantity={self.quantity})>"


class Order(Base):
    """Trade orders."""

    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    quantity: Mapped[Decimal] = mapped_column(Numeric(15, 4), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Order(symbol={self.symbol}, status={self.status})>"


# =============================================================================
# MARKET & ML MODELS
# =============================================================================


class MarketTick(Base):
    """Real-time market ticks."""

    __tablename__ = "market_ticks"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    price: Mapped[Decimal] = mapped_column(Numeric(15, 4), nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<MarketTick(symbol={self.symbol}, price={self.price})>"


class MLModel(Base):
    """Registry of trained ML models."""

    __tablename__ = "ml_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<MLModel(name={self.name}, version={self.version})>"


class ModelPrediction(Base):
    """Predictions generated by ML models."""

    __tablename__ = "model_predictions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    predicted_value: Mapped[float] = mapped_column(Numeric, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric)

    def __repr__(self) -> str:
        return f"<ModelPrediction(symbol={self.symbol}, value={self.predicted_value})>"


# =============================================================================
# RATE LIMIT MODEL
# =============================================================================


class RateLimit(Base):
    """API rate limiting tracking."""

    __tablename__ = "rate_limits"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(100), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, default=1)
