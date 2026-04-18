from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from src.database.base import Base  # Assuming Base is defined in src/database/base.py


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4())) # Using UUID for IDs
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    tier = Column(String, index=True, default="free") # e.g., free, premium, enterprise
    roles = Column(String, server_default="[]", nullable=False) # Store as JSON string or use JSON type if DB supports
    is_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships (if any)
    # portfolios = relationship("Portfolio", back_populates="owner")

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    cash = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    # owner = relationship("User", back_populates="portfolios")
    # trades = relationship("Trade", back_populates="portfolio")

class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, index=True, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    side = Column(String, nullable=False) # 'buy' or 'sell'
    order_type = Column(String, nullable=False) # e.g., 'market', 'limit'
    status = Column(String, index=True, default="pending") # e.g., 'pending', 'filled', 'cancelled'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    # portfolio = relationship("Portfolio", back_populates="trades")

class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True, nullable=False)
    version = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_ml_model_name_version"),)

# Add other models as needed, e.g., for Market Data, Options, etc.
# from .crud import get_user_by_email # Avoid circular imports if crud depends on models and vice-versa during definition
# For now, models are defined independently.

# Ensure UUID is imported if used in default values
import uuid
