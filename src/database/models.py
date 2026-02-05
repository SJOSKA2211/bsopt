"""
SQLAlchemy ORM Models for BSOPT Platform (Neon Optimized)
"""

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
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
    mfa_secret: Mapped[str | None] = mapped_column(String(255))

    # Relationships
    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    oauth_clients: Mapped[list["OAuth2Client"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, tier={self.tier})>"


# =============================================================================
# API KEY MODEL
# =============================================================================


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<APIKey(name={self.name}, prefix={self.key_prefix})>"


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================


class AuditLog(Base):
    __tablename__ = "audit_logs"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, server_default=func.now())
    method: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_ip: Mapped[str] = mapped_column(Text, nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSONB)

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
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# =============================================================================
# OAUTH2 CLIENT MODEL
# =============================================================================


class OAuth2Client(Base):
    """OAuth2 Client Registry."""

    __tablename__ = "oauth2_clients"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    client_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    redirect_uris: Mapped[list[str] | None] = mapped_column(JSONB)
    scopes: Mapped[list[str] | None] = mapped_column(JSONB)
    grant_types: Mapped[list[str] | None] = mapped_column(JSONB)
    response_types: Mapped[list[str] | None] = mapped_column(JSONB)
    is_confidential: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    user: Mapped["User"] = relationship(back_populates="oauth_clients")

    def verify_secret(self, secret: str) -> bool:
        return self.client_secret == secret

    def check_redirect_uri(self, redirect_uri):
        if not self.redirect_uris:
            return False
        return redirect_uri in self.redirect_uris

    def check_client_secret(self, client_secret):
        return self.client_secret == client_secret

    def check_endpoint_auth_method(self, method, endpoint):
        if endpoint == 'token':
            if self.is_confidential:
                return method in ['client_secret_basic', 'client_secret_post']
            return method == 'none'
        return True

    @property
    def client_metadata(self):
        return {
            "redirect_uris": self.redirect_uris,
            "scopes": self.scopes,
            "grant_types": self.grant_types,
            "response_types": self.response_types
        }

    def __repr__(self) -> str:
        return f"<OAuth2Client(id={self.client_id})>"


class OAuth2AuthorizationCode(Base):
    """Temporary codes for Authorization Code flow."""

    __tablename__ = "oauth2_auth_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    client_id: Mapped[str] = mapped_column(String(48))
    redirect_uri: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(Text)
    nonce: Mapped[str | None] = mapped_column(Text)
    auth_time: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))
    code_challenge: Mapped[str | None] = mapped_column(Text)
    code_challenge_method: Mapped[str | None] = mapped_column(String(48))
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    def is_expired(self):
        return self.auth_time + 300 < time.time()


class OAuth2Token(Base):
    """Access and Refresh tokens."""

    __tablename__ = "oauth2_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(48))
    token_type: Mapped[str] = mapped_column(String(40))
    access_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(String(255), index=True)
    scope: Mapped[str] = mapped_column(Text)
    issued_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))
    expires_in: Mapped[int] = mapped_column(Integer, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    def is_expired(self):
        return self.issued_at + self.expires_in < time.time()

    def is_revoked(self):
        return self.revoked


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

    bid: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ask: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    last: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    volume: Mapped[int | None] = mapped_column(Integer)
    open_interest: Mapped[int | None] = mapped_column(Integer)
    implied_volatility: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))

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
    price: Mapped[Decimal | None] = mapped_column(Numeric(15, 4))
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
    volume: Mapped[int | None] = mapped_column(Integer)

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
    metrics: Mapped[dict | None] = mapped_column(JSONB)
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
    confidence: Mapped[float | None] = mapped_column(Numeric)

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


# =============================================================================
# PRICING CALIBRATION MODEL
# =============================================================================


class CalibrationResult(Base):
    """Stores results of pricing model calibration."""

    __tablename__ = "calibration_results"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    v0: Mapped[float] = mapped_column(Numeric, nullable=False)
    kappa: Mapped[float] = mapped_column(Numeric, nullable=False)
    theta: Mapped[float] = mapped_column(Numeric, nullable=False)
    sigma: Mapped[float] = mapped_column(Numeric, nullable=False)
    rho: Mapped[float] = mapped_column(Numeric, nullable=False)
    rmse: Mapped[float] = mapped_column(Numeric, nullable=False)
    r_squared: Mapped[float] = mapped_column(Numeric, nullable=False)
    num_options: Mapped[int] = mapped_column(Integer, nullable=False)
    svi_params: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<CalibrationResult(symbol={self.symbol}, rmse={self.rmse})>"
