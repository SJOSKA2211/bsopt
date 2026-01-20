"""
Optimized CRUD Operations for BSOPT Platform

This module provides:
- Eager loading to prevent N+1 queries
- Bulk operations for performance
- Optimized queries using strategic indexes
- Type-safe operations with proper error handling
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, cast
from uuid import UUID

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.orm import Session, selectinload

from src.security.password import password_service

from .models import (
    MLModel,
    OptionPrice,
    Order,
    Portfolio,
    Position,
    User,
)

# =============================================================================
# USER OPERATIONS
# =============================================================================


def get_user_by_id(db: Session, user_id: UUID) -> Optional[User]:
    """Get user by ID with portfolios eagerly loaded."""
    return db.execute(
        select(User).options(selectinload(User.portfolios)).where(User.id == user_id)
    ).scalar_one_or_none()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email (uses idx_users_email index)."""
    return db.execute(select(User).where(User.email == email)).scalar_one_or_none()


def create_user(db: Session, email: str, password: str, full_name: str) -> User:
    """Create a new user with a verification token."""
    hashed_password = password_service.hash_password(password)
    verification_token = (
        password_service.generate_reset_token()
    )  # Re-use for secure token generation

    db_user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        verification_token=verification_token,
        is_verified=False,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_with_portfolios(db: Session, user_id: UUID) -> Optional[User]:
    """
    Get user with all portfolios and positions eagerly loaded.
    Eliminates N+1 query pattern for dashboard views.
    """
    return db.execute(
        select(User)
        .options(selectinload(User.portfolios).selectinload(Portfolio.positions))
        .where(User.id == user_id)
    ).scalar_one_or_none()


def get_active_users_by_tier(db: Session, tier: str, limit: int = 100) -> Sequence[User]:
    """
    Get active users by tier (uses idx_users_tier_active index).
    """
    return (
        db.execute(
            select(User)
            .where(and_(User.tier == tier, User.is_active.is_(True)))
            .order_by(User.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def update_user_last_login(db: Session, user_id: UUID) -> None:
    """Update user's last login timestamp."""
    db.execute(update(User).where(User.id == user_id).values(last_login=func.now()))
    db.commit()


def update_user_tier(db: Session, user_id: UUID, tier: str) -> None:
    """Update user's subscription tier."""
    db.execute(update(User).where(User.id == user_id).values(tier=tier))
    db.commit()


# =============================================================================
# PORTFOLIO OPERATIONS
# =============================================================================


def get_portfolio_by_id(db: Session, portfolio_id: UUID) -> Optional[Portfolio]:
    """Get portfolio with positions eagerly loaded."""
    return db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.positions))
        .where(Portfolio.id == portfolio_id)
    ).scalar_one_or_none()


def get_portfolio_with_open_positions(db: Session, portfolio_id: UUID) -> Optional[Portfolio]:
    """
    Get portfolio with only open positions.
    Uses idx_positions_portfolio_status_open partial index.
    """
    portfolio = db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.positions))
        .where(Portfolio.id == portfolio_id)
    ).scalar_one_or_none()

    if portfolio:
        # Filter in Python to use cached relationship
        portfolio.positions = [p for p in portfolio.positions if p.status == "open"]

    return portfolio


def get_user_portfolios(
    db: Session, user_id: UUID, include_positions: bool = True
) -> Sequence[Portfolio]:
    """
    Get all portfolios for a user (uses idx_portfolios_user_created index).
    """
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)

    if include_positions:
        stmt = stmt.options(selectinload(Portfolio.positions))

    stmt = stmt.order_by(Portfolio.created_at.desc())

    return db.execute(stmt).scalars().all()


def create_portfolio(
    db: Session, user_id: UUID, name: str, initial_cash: Decimal = Decimal("0.00")
) -> Portfolio:
    """Create a new portfolio for a user."""
    portfolio = Portfolio(user_id=user_id, name=name, cash_balance=initial_cash)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def update_portfolio_cash(
    db: Session, portfolio_id: UUID, amount: Decimal, operation: str = "add"
) -> bool:
    """
    Update portfolio cash balance atomically.
    operation: "add" or "subtract"
    """
    if operation == "add":
        stmt = (
            update(Portfolio)
            .where(Portfolio.id == portfolio_id)
            .values(cash_balance=Portfolio.cash_balance + amount)
        )
    else:
        stmt = (
            update(Portfolio)
            .where(
                and_(
                    Portfolio.id == portfolio_id,
                    Portfolio.cash_balance >= amount,  # Prevent negative
                )
            )
            .values(cash_balance=Portfolio.cash_balance - amount)
        )

    result = cast(Any, db.execute(stmt))
    db.commit()
    return bool(result.rowcount > 0)


# =============================================================================
# POSITION OPERATIONS
# =============================================================================


def get_position_by_id(db: Session, position_id: UUID) -> Optional[Position]:
    """Get position by ID."""
    return db.execute(select(Position).where(Position.id == position_id)).scalar_one_or_none()


def get_open_positions_by_portfolio(db: Session, portfolio_id: UUID) -> Sequence[Position]:
    """
    Get all open positions for a portfolio.
    Uses idx_positions_portfolio_status_open partial index.
    """
    return (
        db.execute(
            select(Position)
            .where(and_(Position.portfolio_id == portfolio_id, Position.status == "open"))
            .order_by(Position.entry_date.desc())
        )
        .scalars()
        .all()
    )


def get_expiring_positions(db: Session, days_until_expiry: int = 7) -> Sequence[Position]:
    """
    Get positions expiring soon across all portfolios.
    Uses idx_positions_expiring_soon partial index.
    """
    expiry_threshold = date.today() + timedelta(days=days_until_expiry)

    return (
        db.execute(
            select(Position)
            .where(
                and_(
                    Position.status == "open",
                    Position.expiry <= expiry_threshold,
                    Position.expiry.isnot(None),
                )
            )
            .order_by(Position.expiry)
        )
        .scalars()
        .all()
    )


def get_positions_by_symbol(
    db: Session, symbol: str, status: Optional[str] = None
) -> Sequence[Position]:
    """
    Get all positions for a symbol.
    Uses idx_positions_symbol_status index.
    """
    stmt = select(Position).where(Position.symbol == symbol)

    if status:
        stmt = stmt.where(Position.status == status)

    return db.execute(stmt.order_by(Position.entry_date.desc())).scalars().all()


def create_position(
    db: Session,
    portfolio_id: UUID,
    symbol: str,
    quantity: int,
    entry_price: Decimal,
    strike: Optional[Decimal] = None,
    expiry: Optional[date] = None,
    option_type: Optional[str] = None,
) -> Position:
    """Create a new position."""
    position = Position(
        portfolio_id=portfolio_id,
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        strike=strike,
        expiry=expiry,
        option_type=option_type,
    )
    db.add(position)
    db.commit()
    db.refresh(position)
    return position


def bulk_create_positions(db: Session, positions_data: List[dict]) -> int:
    """
    Bulk insert positions using SQLAlchemy Core.
    10-50x faster than individual inserts.

    Args:
        positions_data: List of dicts with position fields

    Returns:
        Number of positions created
    """
    if not positions_data:
        return 0

    db.execute(insert(Position), positions_data)
    db.commit()
    return len(positions_data)


def close_position(
    db: Session, position_id: UUID, exit_price: Decimal, exit_date: Optional[datetime] = None
) -> Optional[Position]:
    """Close a position and calculate realized P&L."""
    # Ensure timezone is imported for datetime.now(timezone.utc)
    from datetime import timezone
    position = get_position_by_id(db, position_id)

    if not position or position.status == "closed":
        return None

    position.exit_price = exit_price
    position.exit_date = exit_date or datetime.now(timezone.utc) # Use timezone-aware datetime
    position.status = "closed"
    position.realized_pnl = (exit_price - position.entry_price) * position.quantity

    db.commit()
    db.refresh(position)
    return position


# =============================================================================
# ORDER OPERATIONS
# =============================================================================


def get_order_by_id(db: Session, order_id: UUID) -> Optional[Order]:
    """Get order by ID."""
    return db.execute(select(Order).where(Order.id == order_id)).scalar_one_or_none()


def get_user_orders(
    db: Session, user_id: UUID, status: Optional[str] = None, limit: int = 100
) -> Sequence[Order]:
    """
    Get recent orders for a user.
    Uses idx_orders_user_created index.
    """
    stmt = select(Order).where(Order.user_id == user_id)

    if status:
        stmt = stmt.where(Order.status == status)

    return db.execute(stmt.order_by(Order.created_at.desc()).limit(limit)).scalars().all()


def get_pending_orders(db: Session, limit: int = 1000) -> Sequence[Order]:
    """
    Get all pending orders for processing.
    Uses idx_orders_pending partial index.
    """
    return (
        db.execute(
            select(Order).where(Order.status == "pending").order_by(Order.created_at).limit(limit)
        )
        .scalars()
        .all()
    )


def get_orders_by_broker(
    db: Session, broker: str, broker_order_id: Optional[str] = None
) -> Sequence[Order]:
    """
    Get orders by broker (uses idx_orders_broker_lookup index).
    """
    stmt = select(Order).where(Order.broker == broker)

    if broker_order_id:
        stmt = stmt.where(Order.broker_order_id == broker_order_id)

    return db.execute(stmt).scalars().all()


def create_order(
    db: Session,
    user_id: UUID,
    portfolio_id: UUID,
    symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    limit_price: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None,
    strike: Optional[Decimal] = None,
    expiry: Optional[date] = None,
    option_type: Optional[str] = None,
) -> Order:
    """Create a new order."""
    order = Order(
        user_id=user_id,
        portfolio_id=portfolio_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        strike=strike,
        expiry=expiry,
        option_type=option_type,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def update_order_status(
    db: Session,
    order_id: UUID,
    status: str,
    filled_quantity: Optional[int] = None,
    filled_price: Optional[Decimal] = None,
) -> bool:
    """Update order status and fill information."""
    values: Dict[str, Any] = {"status": status}

    if filled_quantity is not None:
        values["filled_quantity"] = filled_quantity
    if filled_price is not None:
        values["filled_price"] = filled_price

    result = cast(Any, db.execute(update(Order).where(Order.id == order_id).values(**values)))
    db.commit()
    return bool(result.rowcount > 0)


# =============================================================================
# ML MODEL OPERATIONS
# =============================================================================


def get_production_model(db: Session, name: str) -> Optional[MLModel]:
    """
    Get the production version of a model.
    Uses idx_ml_models_production index.
    """
    return db.execute(
        select(MLModel).where(and_(MLModel.name == name, MLModel.is_production.is_(True)))
    ).scalar_one_or_none()


def get_latest_model_version(db: Session, name: str) -> Optional[MLModel]:
    """
    Get the latest version of a model.
    Uses idx_ml_models_version index.
    """
    return db.execute(
        select(MLModel).where(MLModel.name == name).order_by(MLModel.version.desc()).limit(1)
    ).scalar_one_or_none()


def create_model(
    db: Session,
    name: str,
    algorithm: str,
    created_by: Optional[UUID] = None,
    hyperparameters: Optional[dict] = None,
    training_metrics: Optional[dict] = None,
    model_artifact_url: Optional[str] = None,
) -> MLModel:
    """Create a new model version."""
    # Get next version number
    latest = get_latest_model_version(db, name)
    version = (latest.version + 1) if latest else 1

    model = MLModel(
        name=name,
        algorithm=algorithm,
        version=version,
        created_by=created_by,
        hyperparameters=hyperparameters,
        training_metrics=training_metrics,
        model_artifact_url=model_artifact_url,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def set_production_model(db: Session, model_id: UUID) -> bool:
    """Set a model as the production version (unsets others)."""
    model = db.execute(select(MLModel).where(MLModel.id == model_id)).scalar_one_or_none()

    if not model:
        return False

    # Unset previous production model
    db.execute(
        update(MLModel)
        .where(and_(MLModel.name == model.name, MLModel.is_production.is_(True)))
        .values(is_production=False)
    )

    # Set new production model
    model.is_production = True
    db.commit()
    return True


# =============================================================================
# OPTION PRICE OPERATIONS
# =============================================================================


def get_latest_option_price(
    db: Session, symbol: str, strike: Decimal, expiry: date, option_type: str
) -> Optional[OptionPrice]:
    """
    Get the most recent price for an option.
    Uses idx_options_prices_chain index.
    """
    return db.execute(
        select(OptionPrice)
        .where(
            and_(
                OptionPrice.symbol == symbol,
                OptionPrice.strike == strike,
                OptionPrice.expiry == expiry,
                OptionPrice.option_type == option_type,
            )
        )
        .order_by(OptionPrice.time.desc())
        .limit(1)
    ).scalar_one_or_none()


def get_option_chain(
    db: Session, symbol: str, expiry: date, option_type: Optional[str] = None
) -> Sequence[OptionPrice]:
    """
    Get full option chain for a symbol and expiry.
    Uses idx_options_prices_chain_lookup index.
    """
    stmt = select(OptionPrice).where(
        and_(OptionPrice.symbol == symbol, OptionPrice.expiry == expiry)
    )

    if option_type:
        stmt = stmt.where(OptionPrice.option_type == option_type)

    # Get latest price for each strike
    stmt = stmt.order_by(OptionPrice.strike, OptionPrice.time.desc()).distinct(OptionPrice.strike)

    return db.execute(stmt).scalars().all()


def bulk_insert_option_prices(db: Session, prices_data: List[dict]) -> int:
    """
    Bulk insert option prices (for market data ingestion).
    Uses ON CONFLICT DO UPDATE for upsert behavior.
    """
    if not prices_data:
        return 0

    # Define the upsert statement
    insert_stmt = postgresql_insert(OptionPrice).values(prices_data)
    
    # Define the columns to update on conflict (all price and Greek columns)
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[OptionPrice.time, OptionPrice.symbol, OptionPrice.strike, OptionPrice.expiry, OptionPrice.option_type],
        set_=dict(
            bid=insert_stmt.excluded.bid,
            ask=insert_stmt.excluded.ask,
            last=insert_stmt.excluded.last,
            volume=insert_stmt.excluded.volume,
            open_interest=insert_stmt.excluded.open_interest,
            implied_volatility=insert_stmt.excluded.implied_volatility,
            delta=insert_stmt.excluded.delta,
            gamma=insert_stmt.excluded.gamma,
            vega=insert_stmt.excluded.vega,
            theta=insert_stmt.excluded.theta,
            rho=insert_stmt.excluded.rho,
        ),
        # Using a WHERE clause in on_conflict_do_update is generally for conditional updates.
        # For TimescaleDB, we might not need a WHERE clause if the primary key
        # (time, symbol, strike, expiry, option_type) is the only conflict target.
        # I'll remove the example WHERE clause for clarity.
    )

    db.execute(on_conflict_stmt)
    db.commit()
    return len(prices_data)


# =============================================================================
# AGGREGATION QUERIES
# =============================================================================


def get_portfolio_summary(db: Session, portfolio_id: UUID) -> dict:
    """
    Get aggregated portfolio statistics.
    Consider using materialized view for better performance.
    """
    result = cast(
        Any,
        db.execute(
            select(
                func.count(Position.id).filter(Position.status == "open").label("open_positions"),
                func.count(Position.id)
                .filter(Position.status == "closed")
                .label("closed_positions"),
                func.coalesce(
                    func.sum(Position.realized_pnl).filter(Position.status == "closed"), 0
                ).label("total_realized_pnl"),
                func.coalesce(
                    func.sum(Position.quantity * Position.entry_price).filter(
                        Position.status == "open"
                    ),
                    0,
                ).label("open_position_value"),
            ).where(Position.portfolio_id == portfolio_id)
        ).first(),
    )

    portfolio = get_portfolio_by_id(db, portfolio_id)

    return {
        "portfolio_id": portfolio_id,
        "name": portfolio.name if portfolio else None,
        "cash_balance": float(portfolio.cash_balance) if portfolio else 0,
        "open_positions": result.open_positions if result else 0,
        "closed_positions": result.closed_positions if result else 0,
        "total_realized_pnl": float(result.total_realized_pnl or 0) if result else 0.0,
        "open_position_value": float(result.open_position_value or 0) if result else 0.0,
        "total_value": (
            float((portfolio.cash_balance if portfolio else 0) + (result.open_position_value or 0))
            if result
            else float(portfolio.cash_balance) if portfolio else 0.0
        ),
    }


def get_user_trading_stats(db: Session, user_id: UUID) -> dict:
    """Get trading statistics for a user."""
    orders_result = cast(
        Any,
        db.execute(
            select(
                func.count(Order.id).label("total_orders"),
                func.count(Order.id).filter(Order.status == "filled").label("filled_orders"),
                func.count(Order.id).filter(Order.status == "cancelled").label("cancelled_orders"),
                func.avg(Order.filled_price)
                .filter(Order.status == "filled")
                .label("avg_fill_price"),
            ).where(Order.user_id == user_id)
        ).first(),
    )

    if not orders_result:
        return {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "fill_rate": 0,
            "avg_fill_price": 0.0,
        }

    return {
        "total_orders": orders_result.total_orders or 0,
        "filled_orders": orders_result.filled_orders or 0,
        "cancelled_orders": orders_result.cancelled_orders or 0,
        "fill_rate": (
            orders_result.filled_orders / orders_result.total_orders * 100
            if orders_result.total_orders
            else 0
        ),
        "avg_fill_price": float(orders_result.avg_fill_price or 0),
    }
