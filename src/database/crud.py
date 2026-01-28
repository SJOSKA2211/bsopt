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

from anyio.to_thread import run_sync
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

from sqlalchemy.ext.asyncio import AsyncSession

# =============================================================================
# USER OPERATIONS
# =============================================================================


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get user by ID with portfolios eagerly loaded (Async)."""
    result = await db.execute(
        select(User).options(selectinload(User.portfolios)).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email (Async)."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, password: str, full_name: str) -> User:
    """Create a new user (Async)."""
    hashed_password = await run_sync(password_service.hash_password, password)
    verification_token = password_service.generate_reset_token()

    db_user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        verification_token=verification_token,
        is_verified=False,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def get_user_with_portfolios(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get user with all portfolios and positions eagerly loaded (Async)."""
    result = await db.execute(
        select(User)
        .options(selectinload(User.portfolios).selectinload(Portfolio.positions))
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_active_users_by_tier(db: AsyncSession, tier: str, limit: int = 100) -> Sequence[User]:
    """Get active users by tier (Async)."""
    result = await db.execute(
        select(User)
        .where(and_(User.tier == tier, User.is_active.is_(True)))
        .order_by(User.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def update_user_last_login(db: AsyncSession, user_id: UUID) -> None:
    """Update user's last login timestamp (Async)."""
    await db.execute(update(User).where(User.id == user_id).values(last_login=func.now()))
    await db.commit()


async def update_user_tier(db: AsyncSession, user_id: UUID, tier: str) -> None:
    """Update user's subscription tier (Async)."""
    await db.execute(update(User).where(User.id == user_id).values(tier=tier))
    await db.commit()


# =============================================================================
# PORTFOLIO OPERATIONS
# =============================================================================


async def get_portfolio_by_id(db: AsyncSession, portfolio_id: UUID) -> Optional[Portfolio]:
    """Get portfolio with positions eagerly loaded (Async)."""
    result = await db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.positions))
        .where(Portfolio.id == portfolio_id)
    )
    return result.scalar_one_or_none()


async def get_portfolio_with_open_positions(db: AsyncSession, portfolio_id: UUID) -> Optional[Portfolio]:
    """Get portfolio with only open positions (Async)."""
    # Use filtered selectinload to only fetch open positions from the DB
    result = await db.execute(
        select(Portfolio)
        .options(
            selectinload(Portfolio.positions).and_(Position.status == "open")
        )
        .where(Portfolio.id == portfolio_id)
    )
    return result.scalar_one_or_none()


async def get_user_portfolios(
    db: AsyncSession, user_id: UUID, include_positions: bool = True
) -> Sequence[Portfolio]:
    """Get all portfolios for a user (Async)."""
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)

    if include_positions:
        stmt = stmt.options(selectinload(Portfolio.positions))

    stmt = stmt.order_by(Portfolio.created_at.desc())

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_portfolio(
    db: AsyncSession, user_id: UUID, name: str, initial_cash: Decimal = Decimal("0.00")
) -> Portfolio:
    """Create a new portfolio (Async)."""
    portfolio = Portfolio(user_id=user_id, name=name, cash_balance=initial_cash)
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return portfolio


async def update_portfolio_cash(
    db: AsyncSession, portfolio_id: UUID, amount: Decimal, operation: str = "add"
) -> bool:
    """Update portfolio cash balance atomically (Async)."""
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
                    Portfolio.cash_balance >= amount,
                )
            )
            .values(cash_balance=Portfolio.cash_balance - amount)
        )

    result = await db.execute(stmt)
    await db.commit()
    return bool(result.rowcount > 0)


# =============================================================================
# POSITION OPERATIONS
# =============================================================================


async def get_position_by_id(db: AsyncSession, position_id: UUID) -> Optional[Position]:
    """Get position by ID (Async)."""
    result = await db.execute(select(Position).where(Position.id == position_id))
    return result.scalar_one_or_none()


async def get_open_positions_by_portfolio(db: AsyncSession, portfolio_id: UUID) -> Sequence[Position]:
    """Get all open positions for a portfolio (Async)."""
    result = await db.execute(
        select(Position)
        .where(and_(Position.portfolio_id == portfolio_id, Position.status == "open"))
        .order_by(Position.entry_date.desc())
    )
    return result.scalars().all()


async def get_expiring_positions(db: AsyncSession, days_until_expiry: int = 7) -> Sequence[Position]:
    """Get positions expiring soon (Async)."""
    expiry_threshold = date.today() + timedelta(days=days_until_expiry)

    result = await db.execute(
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
    return result.scalars().all()


async def get_positions_by_symbol(
    db: AsyncSession, symbol: str, status: Optional[str] = None
) -> Sequence[Position]:
    """Get all positions for a symbol (Async)."""
    stmt = select(Position).where(Position.symbol == symbol)

    if status:
        stmt = stmt.where(Position.status == status)

    result = await db.execute(stmt.order_by(Position.entry_date.desc()))
    return result.scalars().all()


async def create_position(
    db: AsyncSession,
    portfolio_id: UUID,
    symbol: str,
    quantity: int,
    entry_price: Decimal,
    strike: Optional[Decimal] = None,
    expiry: Optional[date] = None,
    option_type: Optional[str] = None,
) -> Position:
    """Create a new position (Async)."""
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
    await db.commit()
    await db.refresh(position)
    return position


async def bulk_create_positions(db: AsyncSession, positions_data: List[dict]) -> int:
    """Bulk insert positions (Async)."""
    if not positions_data:
        return 0

    await db.execute(insert(Position), positions_data)
    await db.commit()
    return len(positions_data)


async def close_position(
    db: AsyncSession, position_id: UUID, exit_price: Decimal, exit_date: Optional[datetime] = None
) -> Optional[Position]:
    """Close a position (Async)."""
    from datetime import timezone
    position = await get_position_by_id(db, position_id)

    if not position or position.status == "closed":
        return None

    position.exit_price = exit_price
    position.exit_date = exit_date or datetime.now(timezone.utc)
    position.status = "closed"
    position.realized_pnl = (exit_price - position.entry_price) * position.quantity

    await db.commit()
    await db.refresh(position)
    return position


# =============================================================================
# ORDER OPERATIONS
# =============================================================================


async def get_order_by_id(db: AsyncSession, order_id: UUID) -> Optional[Order]:
    """Get order by ID (Async)."""
    result = await db.execute(select(Order).where(Order.id == order_id))
    return result.scalar_one_or_none()


async def get_user_orders(
    db: AsyncSession, user_id: UUID, status: Optional[str] = None, limit: int = 100
) -> Sequence[Order]:
    """Get recent orders for a user (Async)."""
    stmt = select(Order).where(Order.user_id == user_id)

    if status:
        stmt = stmt.where(Order.status == status)

    result = await db.execute(stmt.order_by(Order.created_at.desc()).limit(limit))
    return result.scalars().all()


async def get_pending_orders(db: AsyncSession, limit: int = 1000) -> Sequence[Order]:
    """Get all pending orders (Async)."""
    result = await db.execute(
        select(Order).where(Order.status == "pending").order_by(Order.created_at).limit(limit)
    )
    return result.scalars().all()


async def get_orders_by_broker(
    db: AsyncSession, broker: str, broker_order_id: Optional[str] = None
) -> Sequence[Order]:
    """Get orders by broker (Async)."""
    stmt = select(Order).where(Order.broker == broker)

    if broker_order_id:
        stmt = stmt.where(Order.broker_order_id == broker_order_id)

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_order(
    db: AsyncSession,
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
    """Create a new order (Async)."""
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
    await db.commit()
    await db.refresh(order)
    return order


async def update_order_status(
    db: AsyncSession,
    order_id: UUID,
    status: str,
    filled_quantity: Optional[int] = None,
    filled_price: Optional[Decimal] = None,
) -> bool:
    """Update order status (Async)."""
    values: Dict[str, Any] = {"status": status}

    if filled_quantity is not None:
        values["filled_quantity"] = filled_quantity
    if filled_price is not None:
        values["filled_price"] = filled_price

    result = await db.execute(update(Order).where(Order.id == order_id).values(**values))
    await db.commit()
    return bool(result.rowcount > 0)


# =============================================================================
# ML MODEL OPERATIONS
# =============================================================================


async def get_production_model(db: AsyncSession, name: str) -> Optional[MLModel]:
    """Get the production version of a model (Async)."""
    result = await db.execute(
        select(MLModel).where(and_(MLModel.name == name, MLModel.is_production.is_(True)))
    )
    return result.scalar_one_or_none()


async def get_latest_model_version(db: AsyncSession, name: str) -> Optional[MLModel]:
    """Get the latest version of a model (Async)."""
    result = await db.execute(
        select(MLModel).where(MLModel.name == name).order_by(MLModel.version.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def create_model(
    db: AsyncSession,
    name: str,
    algorithm: str,
    created_by: Optional[UUID] = None,
    hyperparameters: Optional[dict] = None,
    training_metrics: Optional[dict] = None,
    model_artifact_url: Optional[str] = None,
) -> MLModel:
    """Create a new model version (Async)."""
    latest = await get_latest_model_version(db, name)
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
    await db.commit()
    await db.refresh(model)
    return model


async def set_production_model(db: AsyncSession, model_id: UUID) -> bool:
    """Set a model as production (Async)."""
    result = await db.execute(select(MLModel).where(MLModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        return False

    await db.execute(
        update(MLModel)
        .where(and_(MLModel.name == model.name, MLModel.is_production.is_(True)))
        .values(is_production=False)
    )

    model.is_production = True
    await db.commit()
    return True


# =============================================================================
# OPTION PRICE OPERATIONS
# =============================================================================


async def get_latest_option_price(
    db: AsyncSession, symbol: str, strike: Decimal, expiry: date, option_type: str
) -> Optional[OptionPrice]:
    """Get the most recent price for an option (Async)."""
    result = await db.execute(
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
    )
    return result.scalar_one_or_none()


async def get_option_chain(
    db: AsyncSession, symbol: str, expiry: date, option_type: Optional[str] = None
) -> Sequence[OptionPrice]:
    """Get full option chain (Async)."""
    stmt = select(OptionPrice).where(
        and_(OptionPrice.symbol == symbol, OptionPrice.expiry == expiry)
    )

    if option_type:
        stmt = stmt.where(OptionPrice.option_type == option_type)

    stmt = stmt.order_by(OptionPrice.strike, OptionPrice.time.desc()).distinct(OptionPrice.strike)

    result = await db.execute(stmt)
    return result.scalars().all()


async def bulk_insert_option_prices(db: AsyncSession, prices_data: List[dict]) -> int:
    """Bulk insert option prices (Async)."""
    if not prices_data:
        return 0

    insert_stmt = postgresql_insert(OptionPrice).values(prices_data)
    
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
    )

    await db.execute(on_conflict_stmt)
    await db.commit()
    return len(prices_data)


# =============================================================================
# AGGREGATION QUERIES
# =============================================================================


async def get_portfolio_summary(db: AsyncSession, portfolio_id: UUID) -> dict:
    """Get aggregated portfolio statistics (Async)."""
    result_set = await db.execute(
        select(
            func.count(Position.id).filter(Position.status == "open").label("open_positions"),
            func.count(Position.id).filter(Position.status == "closed").label("closed_positions"),
            func.coalesce(
                func.sum(Position.realized_pnl).filter(Position.status == "closed"), 0
            ).label("total_realized_pnl"),
            func.coalesce(
                func.sum(Position.quantity * Position.entry_price).filter(Position.status == "open"),
                0,
            ).label("open_position_value"),
        ).where(Position.portfolio_id == portfolio_id)
    )
    result = result_set.first()

    portfolio = await get_portfolio_by_id(db, portfolio_id)

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
            if result and portfolio
            else float(portfolio.cash_balance) if portfolio else 0.0
        ),
    }


async def get_user_trading_stats(db: AsyncSession, user_id: UUID) -> dict:
    """Get trading statistics for a user (Async)."""
    result_set = await db.execute(
        select(
            func.count(Order.id).label("total_orders"),
            func.count(Order.id).filter(Order.status == "filled").label("filled_orders"),
            func.count(Order.id).filter(Order.status == "cancelled").label("cancelled_orders"),
            func.avg(Order.filled_price).filter(Order.status == "filled").label("avg_fill_price"),
        ).where(Order.user_id == user_id)
    )
    orders_result = result_set.first()

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
