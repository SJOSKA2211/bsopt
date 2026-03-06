"""
SQLAlchemy ORM Models for BSOPT Platform (Optimized for PG16 + TimescaleDB)
"""

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# CUSTOM TYPES (Synced with DB ENUMs)

UserTier = ENUM("free", "pro", "enterprise", name="user_tier", create_type=False)
OrderSide = ENUM("buy", "sell", name="order_side", create_type=False)
OrderStatus = ENUM(
    "pending",
    "filled",
    "partially_filled",
    "cancelled",
    "rejected",
    name="order_status",
    create_type=False,
)
OrderType = ENUM("market", "limit", "stop", "stop_limit", name="order_type", create_type=False)
PositionStatus = ENUM("open", "closed", name="position_status", create_type=False)
OptionType = ENUM("call", "put", name="option_type", create_type=False)
MLAlgorithm = ENUM(
    "xgboost",
    "lightgbm",
    "neural_network",
    "random_forest",
    "svm",
    "ensemble",
    name="ml_algorithm",
    create_type=False,
)


# USER MODEL


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(UserTier, default="free")
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
    api_keys: Mapped[list["APIKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    better_auth_sessions: Mapped[list["BetterAuthSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    better_auth_accounts: Mapped[list["BetterAuthAccount"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    method: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True))
    client_ip: Mapped[str] = mapped_column(Text, nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Double, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSONB)


class RequestLog(Base):
    __tablename__ = "request_logs"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    status_code: Mapped[int | None] = mapped_column(Integer)
    path: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[float | None] = mapped_column(Double)


# PORTFOLIO & TRADING


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
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    exit_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    status: Mapped[str] = mapped_column(PositionStatus, default="open")
    strike: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    expiry: Mapped[date | None] = mapped_column(Date)
    option_type: Mapped[str | None] = mapped_column(OptionType)

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
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    strike: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    expiry: Mapped[date | None] = mapped_column(Date)
    option_type: Mapped[str | None] = mapped_column(OptionType)
    side: Mapped[str] = mapped_column(OrderSide, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    order_type: Mapped[str] = mapped_column(OrderType, nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    status: Mapped[str] = mapped_column(OrderStatus, default="pending")
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    broker: Mapped[str | None] = mapped_column(String(50))
    broker_order_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# MARKET DATA (Hypertables)


class OptionPrice(Base):
    __tablename__ = "options_prices"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    strike: Mapped[Decimal] = mapped_column(Numeric(12, 2), primary_key=True)
    expiry: Mapped[date] = mapped_column(Date, primary_key=True)
    option_type: Mapped[str] = mapped_column(OptionType, primary_key=True)

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
    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    price: Mapped[Decimal] = mapped_column(Numeric(15, 4), nullable=False)
    volume: Mapped[int | None] = mapped_column(Integer)
    side: Mapped[str | None] = mapped_column(String(4))


# ML & PREDICTIONS


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(MLAlgorithm, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    training_metrics: Mapped[dict | None] = mapped_column(JSONB)
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
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    input_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
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
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# OAUTH & SECURITY


class OAuth2Client(Base):
    __tablename__ = "oauth2_clients"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    client_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    redirect_uris: Mapped[list[str] | None] = mapped_column(JSONB)
    scopes: Mapped[list[str] | None] = mapped_column(JSONB)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    user: Mapped["User"] = relationship(back_populates="oauth_clients")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")


class BetterAuthSession(Base):
    __tablename__ = "better_auth_sessions"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_agent: Mapped[str | None] = mapped_column(Text)
    ip_address: Mapped[str | None] = mapped_column(String(45))

    user: Mapped["User"] = relationship(back_populates="better_auth_sessions")


class BetterAuthAccount(Base):
    __tablename__ = "better_auth_accounts"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    account_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(255), nullable=False)
    access_token: Mapped[str | None] = mapped_column(Text)
    refresh_token: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password: Mapped[str | None] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="better_auth_accounts")


class SecurityIncident(Base):
    __tablename__ = "security_incidents"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    nature_of_breach: Mapped[str | None] = mapped_column(Text)
    approximate_number_data_subjects: Mapped[int | None] = mapped_column(Integer)
    approximate_number_records: Mapped[int | None] = mapped_column(Integer)
    likely_consequences: Mapped[str | None] = mapped_column(Text)
    measures_taken: Mapped[str | None] = mapped_column(Text)
    data_categories_affected: Mapped[list[str] | None] = mapped_column(JSONB)
    reported_to_dpa: Mapped[bool] = mapped_column(Boolean, default=False)
    reported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OptionContract(Base):
    __tablename__ = "option_contracts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    underlying: Mapped[str] = mapped_column(String, nullable=False)
    expiry: Mapped[date] = mapped_column(Date, nullable=False)
    strike: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    option_type: Mapped[str] = mapped_column(OptionType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("underlying", "expiry", "strike", "option_type"),)


class RLEpisode(Base):
    __tablename__ = "rl_episodes"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    episode_reward: Mapped[float] = mapped_column(Double, nullable=False)
    steps: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


try:
    from pgvector.sqlalchemy import Vector

    class ModelEmbedding(Base):
        __tablename__ = "model_embeddings"

        id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        model_id: Mapped[UUID] = mapped_column(
            ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False
        )
        version: Mapped[int] = mapped_column(Integer, nullable=False)
        embedding: Mapped[list[float]] = mapped_column(Vector(1536))
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), server_default=func.now()
        )


class RateLimit(Base):
    __tablename__ = "rate_limits"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(255), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, default=1)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource: Mapped[str | None] = mapped_column(String(100))
    payload: Mapped[dict | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

except ImportError:
    # Fallback if pgvector not installed
    class ModelEmbedding(Base):  # type: ignore
        __tablename__ = "model_embeddings"

        id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        model_id: Mapped[UUID] = mapped_column(
            ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False
        )
        version: Mapped[int] = mapped_column(Integer, nullable=False)
        embedding: Mapped[dict] = mapped_column(JSONB)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), server_default=func.now()
        )


class RateLimit(Base):
    __tablename__ = "rate_limits"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(255), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, default=1)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource: Mapped[str | None] = mapped_column(String(100))
    payload: Mapped[dict | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
