"""
SQLAlchemy ORM Models for BSOPT Platform (Optimized for PG16 + TimescaleDB)
"""

import time
from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Double,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# =============================================================================
# USER MODEL
# =============================================================================


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(String(20), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_token: Mapped[str | None] = mapped_column(String(255))
    reset_token: Mapped[str | None] = mapped_column(String(255))
    reset_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[str | None] = mapped_column(String(255))
    mfa_backup_codes: Mapped[str | None] = mapped_column(Text)

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    oauth_clients: Mapped[list["OAuth2Client"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


# =============================================================================
# AUDIT & LOGGING (Hypertables)
# =============================================================================


class AuditLog(Base):
    __tablename__ = "audit_logs"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    method: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_ip: Mapped[str] = mapped_column(Text, nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Double, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON)


class RequestLog(Base):
    __tablename__ = "request_logs"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    status_code: Mapped[int | None] = mapped_column(Integer)
    path: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[float | None] = mapped_column(Double)


# =============================================================================
# PORTFOLIO & TRADING
# =============================================================================


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=Decimal("0.00"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="portfolios")
    positions: Mapped[list["Position"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("user_id", "name"),)


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    exit_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    status: Mapped[str] = mapped_column(String(10), default="open")
    strike: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    expiry: Mapped[date | None] = mapped_column(Date)
    option_type: Mapped[str | None] = mapped_column(String(4))

    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    portfolio_id: Mapped[UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    strike: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    expiry: Mapped[date | None] = mapped_column(Date)
    option_type: Mapped[str | None] = mapped_column(String(4))
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    order_type: Mapped[str] = mapped_column(String(15), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    broker: Mapped[str | None] = mapped_column(String(50))
    broker_order_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# =============================================================================
# MARKET DATA (Hypertables)
# =============================================================================


class OptionPrice(Base):
    __tablename__ = "options_prices"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    strike: Mapped[Decimal] = mapped_column(Numeric(12, 2), primary_key=True)
    expiry: Mapped[date] = mapped_column(Date, primary_key=True)
    option_type: Mapped[str] = mapped_column(String(4), primary_key=True)

    bid: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ask: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    last: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    volume: Mapped[int | None] = mapped_column(Integer)
    open_interest: Mapped[int | None] = mapped_column(Integer)
    
    # OPTIMIZED: Using Double Precision for Greeks
    implied_volatility: Mapped[float | None] = mapped_column(Double)
    delta: Mapped[float | None] = mapped_column(Double)
    gamma: Mapped[float | None] = mapped_column(Double)
    vega: Mapped[float | None] = mapped_column(Double)
    theta: Mapped[float | None] = mapped_column(Double)
    rho: Mapped[float | None] = mapped_column(Double)


class MarketTick(Base):
    __tablename__ = "market_ticks"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    price: Mapped[Decimal] = mapped_column(Numeric(15, 4), nullable=False)
    volume: Mapped[int | None] = mapped_column(Integer)
    side: Mapped[str | None] = mapped_column(String(4))


# =============================================================================
# ML & PREDICTIONS
# =============================================================================


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON)
    training_metrics: Mapped[dict | None] = mapped_column(JSON)
    model_artifact_url: Mapped[str | None] = mapped_column(String(500))
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("name", "version"),)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), default=uuid4)
    model_id: Mapped[UUID | None] = mapped_column(ForeignKey("ml_models.id", ondelete="SET NULL"))
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    actual_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    prediction_error: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    actual_value: Mapped[Decimal | None] = mapped_column(Numeric)


class ModelDriftBaseline(Base):
    __tablename__ = "model_drift_baselines"

    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("ml_models.id", ondelete="CASCADE"), primary_key=True
    )
    baseline_accuracy: Mapped[float | None] = mapped_column(Double)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# =============================================================================
# OAUTH & SECURITY
# =============================================================================


class OAuth2Client(Base):
    __tablename__ = "oauth2_clients"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    client_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    redirect_uris: Mapped[list[str] | None] = mapped_column(JSON)
    scopes: Mapped[list[str] | None] = mapped_column(JSON)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    user: Mapped["User"] = relationship(back_populates="oauth_clients")
