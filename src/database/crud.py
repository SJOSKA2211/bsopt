"""
Optimized CRUD Operations for BSOPT Platform

This module provides:
- Eager loading to prevent N+1 queries
- Bulk operations for performance
- Optimized queries using strategic indexes
- Type-safe operations with proper error handling
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import orjson
import structlog
from sqlalchemy import and_, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import (
    AuditLog,
    MarketTick,
    MLModel,
    OptionPrice,
    Order,
    Portfolio,
    Position,
    RequestLog,
    User,
)

logger = structlog.get_logger()

# USER OPERATIONS


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    """Get user by ID with portfolios eagerly loaded (Async)."""
    result = await db.execute(
        select(User).options(selectinload(User.portfolios)).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Get user by email (Async)."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, password: str, full_name: str) -> User:
    """Create a new user using Native Postgres Procedure (Async)."""
    # OPTIMIZED: Hand off hashing to the database layer
    result = await db.execute(
        text("SELECT register_user_native(:email, :password, :full_name)"),
        {"email": email, "password": password, "full_name": full_name},
    )
    user_id = result.scalar()

    # Retrieve the created user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise ValueError("Failed to retrieve created user")

    return user


async def get_user_with_portfolios(db: AsyncSession, user_id: UUID) -> User | None:
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
    """Update user's last login timestamp using native procedure (Async)."""
    await db.execute(text("SELECT update_last_login_native(:uid)"), {"uid": user_id})
    await db.commit()


async def update_user_tier(db: AsyncSession, user_id: UUID, tier: str) -> None:
    """Update user's subscription tier (Async)."""
    await db.execute(update(User).where(User.id == user_id).values(tier=tier))
    await db.commit()


# PORTFOLIO OPERATIONS


async def get_portfolio_by_id(db: AsyncSession, portfolio_id: UUID) -> Portfolio | None:
    """Get portfolio with positions eagerly loaded (Async)."""
    result = await db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.positions))
        .where(Portfolio.id == portfolio_id)
    )
    return result.scalar_one_or_none()


async def get_portfolio_with_open_positions(
    db: AsyncSession, portfolio_id: UUID
) -> Portfolio | None:
    """Get portfolio with only open positions (Async)."""
    # Use filtered selectinload to only fetch open positions from the DB
    result = await db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.positions).and_(Position.status == "open"))
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


# POSITION OPERATIONS


async def get_position_by_id(db: AsyncSession, position_id: UUID) -> Position | None:
    """Get position by ID (Async)."""
    result = await db.execute(select(Position).where(Position.id == position_id))
    return result.scalar_one_or_none()


async def get_open_positions_by_portfolio(
    db: AsyncSession, portfolio_id: UUID
) -> Sequence[Position]:
    """Get all open positions for a portfolio (Async)."""
    result = await db.execute(
        select(Position)
        .where(and_(Position.portfolio_id == portfolio_id, Position.status == "open"))
        .order_by(Position.entry_date.desc())
    )
    return result.scalars().all()


async def get_expiring_positions(
    db: AsyncSession, days_until_expiry: int = 7
) -> Sequence[Position]:
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
    db: AsyncSession, symbol: str, status: str | None = None
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
    strike: Decimal | None = None,
    expiry: date | None = None,
    option_type: str | None = None,
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


async def bulk_create_positions(db: AsyncSession, positions_data: list[dict]) -> int:
    """
    High-performance bulk insert for Positions using PostgreSQL COPY.
    """
    if not positions_data:
        return 0

    columns = [
        "portfolio_id",
        "symbol",
        "strike",
        "expiry",
        "option_type",
        "quantity",
        "entry_price",
        "entry_date",
        "current_price",
        "exit_price",
        "exit_date",
        "realized_pnl",
        "status",
    ]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            records = [
                (
                    row.get("portfolio_id"),
                    row.get("symbol"),
                    row.get("strike"),
                    row.get("expiry"),
                    row.get("option_type"),
                    row.get("quantity"),
                    row.get("entry_price"),
                    row.get("entry_date"),
                    row.get("current_price"),
                    row.get("exit_price"),
                    row.get("exit_date"),
                    row.get("realized_pnl"),
                    row.get("status"),
                )
                for row in positions_data
            ]

            await driver_conn.copy_records_to_table(
                "positions", records=records, columns=columns, timeout=30
            )
            await db.commit()
            return len(positions_data)
        await db.execute(insert(Position), positions_data)
        await db.commit()
        return len(positions_data)

    except Exception as e:
        logger.error("positions_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


async def close_position(
    db: AsyncSession,
    position_id: UUID,
    exit_price: Decimal,
    exit_date: datetime | None = None,
) -> Position | None:
    """Close a position (Async)."""
    position = await get_position_by_id(db, position_id)

    if not position or position.status == "closed":
        return None

    position.exit_price = exit_price
    position.exit_date = exit_date or datetime.now(UTC)
    position.status = "closed"
    position.realized_pnl = (exit_price - position.entry_price) * position.quantity

    await db.commit()
    await db.refresh(position)
    return position


# ORDER OPERATIONS


async def get_order_by_id(db: AsyncSession, order_id: UUID) -> Order | None:
    """Get order by ID (Async)."""
    result = await db.execute(select(Order).where(Order.id == order_id))
    return result.scalar_one_or_none()


async def get_user_orders(
    db: AsyncSession, user_id: UUID, status: str | None = None, limit: int = 100
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
    db: AsyncSession, broker: str, broker_order_id: str | None = None
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
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
    strike: Decimal | None = None,
    expiry: date | None = None,
    option_type: str | None = None,
) -> Order:
    """Create a new order (Async)."""
    order = Order(
        user_id=user_id,
        portfolio_id=portfolio_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        status="pending",
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
    filled_quantity: int | None = None,
    filled_price: Decimal | None = None,
) -> bool:
    """Update order status (Async)."""
    values: dict[str, Any] = {"status": status}

    if filled_quantity is not None:
        values["filled_quantity"] = filled_quantity
    if filled_price is not None:
        values["filled_price"] = filled_price

    result = await db.execute(update(Order).where(Order.id == order_id).values(**values))
    await db.commit()
    return bool(result.rowcount > 0)


# ML MODEL OPERATIONS


async def get_similar_models(
    db: AsyncSession, embedding: list[float], limit: int = 5
) -> list[dict]:
    """
    High-Performance: Vector Similarity Search (HNSW Optimized).
    Finds models with similar semantic embeddings using cosine distance.
    """
    try:
        #  Use <=> for cosine distance with HNSW index
        stmt = text("""
            SELECT 
                me.model_id, 
                me.version, 
                m.name, 
                m.algorithm,
                (1 - (me.embedding <=> :emb)) as similarity
            FROM model_embeddings me
            JOIN ml_models m ON me.model_id = m.id
            ORDER BY me.embedding <=> :emb
            LIMIT :limit
        """)
        result = await db.execute(stmt, {"emb": str(embedding), "limit": limit})
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("vector_similarity_search_failed", error=str(e))
        return []


async def get_production_model(db: AsyncSession, name: str) -> MLModel | None:
    """Get the production version of a model (Async)."""
    result = await db.execute(
        select(MLModel).where(and_(MLModel.name == name, MLModel.is_production.is_(True)))
    )
    return result.scalar_one_or_none()


async def get_latest_model_version(db: AsyncSession, name: str) -> MLModel | None:
    """Get the latest version of a model (Async)."""
    result = await db.execute(
        select(MLModel).where(MLModel.name == name).order_by(MLModel.version.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def create_model(
    db: AsyncSession,
    name: str,
    algorithm: str,
    model_artifact_url: str,
    created_by: UUID | None = None,
    hyperparameters: dict | None = None,
    training_metrics: dict | None = None,
) -> MLModel:
    """Create a new model version (Async)."""
    latest = await get_latest_model_version(db, name)
    version = (latest.version + 1) if latest else 1

    model = MLModel(
        name=name,
        algorithm=algorithm,
        version=version,
        model_artifact_url=model_artifact_url,
        created_by=created_by,
        hyperparameters=hyperparameters,
        training_metrics=training_metrics,
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


# OPTION PRICE OPERATIONS


async def get_latest_option_price(
    db: AsyncSession, symbol: str, strike: Decimal, expiry: date, option_type: str
) -> OptionPrice | None:
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
    db: AsyncSession, symbol: str, expiry: date, option_type: str | None = None
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


async def bulk_insert_option_prices(db: AsyncSession, prices_data: list[dict]) -> int:
    """
    High-performance bulk insert using PostgreSQL COPY command into a staging table.
    Handles unique constraint conflicts with 'ON CONFLICT DO NOTHING' for robust ingestion.
    """
    if not prices_data:
        return 0

    columns = [
        "time",
        "symbol",
        "strike",
        "expiry",
        "option_type",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            records = [
                (
                    row.get("time"),
                    row.get("symbol"),
                    row.get("strike"),
                    row.get("expiry"),
                    row.get("option_type"),
                    row.get("bid"),
                    row.get("ask"),
                    row.get("last"),
                    row.get("volume"),
                    row.get("open_interest"),
                    row.get("implied_volatility"),
                    row.get("delta"),
                    row.get("gamma"),
                    row.get("vega"),
                    row.get("theta"),
                    row.get("rho"),
                )
                for row in prices_data
            ]

            # OPTIMIZED: Light staging table
            await db.execute(
                text("CREATE TEMP TABLE staging_option_prices (LIKE options_prices) ON COMMIT DROP")
            )

            # 2. Fast COPY into staging
            await driver_conn.copy_records_to_table(
                "staging_option_prices", records=records, columns=columns, timeout=30
            )

            # 3. UPSERT/INSERT from staging to main with conflict resolution
            col_list = ", ".join(columns)
            # nosec B608: col_list is constructed from hardcoded columns list, safe from injection
            # The query uses a staging table to merge data safely.
            query = f"""
                INSERT INTO options_prices ({col_list})
                SELECT {col_list} FROM staging_option_prices
                ON CONFLICT DO NOTHING
            """  # nosec B608
            await db.execute(text(query))

            await db.commit()
            return len(prices_data)
        # [FALLBACK]
        await db.execute(insert(OptionPrice), prices_data)
        await db.commit()
        return len(prices_data)

    except Exception as e:
        logger.error("option_prices_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


async def bulk_insert_market_ticks(db: AsyncSession, ticks_data: list[dict]) -> int:
    """
    Ultra-high performance bulk insert for MarketTicks using PostgreSQL binary COPY.
    Handles potential duplicates gracefully via a staging table and 'ON CONFLICT DO NOTHING'.
    """
    if not ticks_data:
        return 0

    columns = ["time", "symbol", "price", "volume", "market", "change", "side"]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            records = [
                (
                    row.get("time") or datetime.now(UTC),
                    row.get("symbol"),
                    row.get("price"),
                    row.get("volume"),
                    row.get("market"),
                    row.get("change") or 0.0,
                    row.get("side"),
                )
                for row in ticks_data
            ]

            # 1. Create temporary staging table (Lean: No indexes/constraints)
            await db.execute(
                text("CREATE TEMP TABLE staging_market_ticks (LIKE market_ticks) ON COMMIT DROP")
            )

            # 2. Fast COPY
            await driver_conn.copy_records_to_table(
                "staging_market_ticks", records=records, columns=columns, timeout=30
            )

            # 3. Safe Merge
            col_list = ", ".join(columns)
            # nosec B608: col_list is constructed from hardcoded columns list, safe from injection
            # Uses ON CONFLICT DO NOTHING to ignore duplicates during bulk load.
            query = f"""
                INSERT INTO market_ticks ({col_list})
                SELECT {col_list} FROM staging_market_ticks
                ON CONFLICT DO NOTHING
            """  # nosec B608
            await db.execute(text(query))

            await db.commit()
            return len(ticks_data)
        await db.execute(insert(MarketTick), ticks_data)
        await db.commit()
        return len(ticks_data)

    except Exception as e:
        logger.error("market_ticks_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


async def bulk_insert_mesh_data(db: AsyncSession, mesh_data: list[dict]) -> int:
    """
    High-performance bulk insert for Market Data Mesh using PostgreSQL COPY.
    """
    if not mesh_data:
        return 0

    columns = [
        "time",
        "symbol",
        "market",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_type",
        "metadata",
    ]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            import msgspec
            records = [
                (
                    row.get("time") or datetime.now(UTC),
                    row.get("symbol"),
                    row.get("market"),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                    row.get("source_type", "scraper"),
                    (
                        msgspec.json.encode(row.get("metadata", {})).decode("utf-8")
                        if row.get("metadata")
                        else None
                    ),
                )
                for row in mesh_data
            ]

            # Use staging for mesh data as well to ensure idempotency if needed
            await db.execute(
                text("CREATE TEMP TABLE staging_market_mesh (LIKE market_data_mesh) ON COMMIT DROP")
            )

            await driver_conn.copy_records_to_table(
                "staging_market_mesh", records=records, columns=columns, timeout=30
            )

            col_list = ", ".join(columns)
            query = f"""
                INSERT INTO market_data_mesh ({col_list})
                SELECT {col_list} FROM staging_market_mesh
                ON CONFLICT DO NOTHING
            """  # nosec B608
            await db.execute(text(query))

            await db.commit()
            return len(mesh_data)

        # Fallback to standard insert
        from src.database.models import MarketDataMesh

        await db.execute(insert(MarketDataMesh), mesh_data)
        await db.commit()
        return len(mesh_data)

    except Exception as e:
        logger.error("market_mesh_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


async def bulk_insert_audit_logs(db: AsyncSession, logs_data: list[dict]) -> int:
    """
    High-performance bulk insert for AuditLog using PostgreSQL COPY.
    """
    if not logs_data:
        return 0

    columns = [
        "time",
        "method",
        "path",
        "status_code",
        "user_id",
        "client_ip",
        "user_agent",
        "latency_ms",
        "metadata",
    ]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            records = [
                (
                    row.get("time") or datetime.now(UTC),
                    row.get("method"),
                    row.get("path"),
                    row.get("status_code"),
                    row.get("user_id"),
                    row.get("client_ip"),
                    row.get("user_agent"),
                    row.get("latency_ms"),
                    (
                        orjson.dumps(row.get("metadata", {})).decode("utf-8")
                        if row.get("metadata")
                        else None
                    ),
                )
                for row in logs_data
            ]

            await driver_conn.copy_records_to_table(
                "audit_logs", records=records, columns=columns, timeout=30
            )
            await db.commit()
            return len(logs_data)
        await db.execute(insert(AuditLog), logs_data)
        await db.commit()
        return len(logs_data)

    except Exception as e:
        logger.error("audit_logs_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


async def bulk_insert_request_logs(db: AsyncSession, logs_data: list[dict]) -> int:
    """
    High-performance bulk insert for RequestLog using PostgreSQL COPY.
    """
    if not logs_data:
        return 0

    columns = [
        "id",
        "request_id",
        "method",
        "path",
        "query_params",
        "headers",
        "client_ip",
        "user_id",
        "status_code",
        "response_time_ms",
        "error_message",
        "created_at",
    ]

    try:
        conn = await db.connection()
        raw_conn = await conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        if hasattr(driver_conn, "copy_records_to_table"):
            records = [
                (
                    row.get("id") or uuid4(),
                    row.get("request_id"),
                    row.get("method"),
                    row.get("path"),
                    row.get("query_params"),
                    (
                        orjson.dumps(row.get("headers", {})).decode("utf-8")
                        if row.get("headers")
                        else None
                    ),
                    row.get("client_ip"),
                    row.get("user_id"),
                    row.get("status_code"),
                    row.get("response_time_ms"),
                    row.get("error_message"),
                    row.get("created_at", datetime.now(UTC)),
                )
                for row in logs_data
            ]

            await driver_conn.copy_records_to_table(
                "request_logs", records=records, columns=columns, timeout=30
            )
            await db.commit()
            return len(logs_data)
        await db.execute(insert(RequestLog), logs_data)
        await db.commit()
        return len(logs_data)

    except Exception as e:
        logger.error("request_logs_bulk_copy_failed", error=str(e))
        await db.rollback()
        raise


# AGGREGATION QUERIES


async def get_portfolio_summary(db: AsyncSession, user_id: UUID) -> list[dict]:
    """
    Get portfolio summary using the optimized materialized view.
    """
    try:
        result = await db.execute(
            text("SELECT * FROM portfolio_summary_mv WHERE user_id = :uid"),
            {"uid": user_id},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.warning("mv_query_failed_falling_back_to_realtime", error=str(e))
        # Fallback logic would go here if MV is not available
        return []


async def get_user_trading_stats(db: AsyncSession, user_id: UUID) -> dict:
    """
    Get trading statistics for a user using the optimized materialized view.
    """
    try:
        result = await db.execute(
            text("SELECT * FROM trading_stats_mv WHERE user_id = :uid"),
            {"uid": user_id},
        )
        row = result.first()

        if not row:
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "cancelled_orders": 0,
                "fill_rate": 0,
                "avg_fill_price": 0.0,
            }

        data = dict(row._mapping)
        total = data["total_orders"]
        filled = data["filled_orders"]

        return {
            "total_orders": total,
            "filled_orders": filled,
            "cancelled_orders": data["cancelled_orders"],
            "fill_rate": (filled / total * 100) if total > 0 else 0,
            "avg_fill_price": float(data["avg_fill_price"] or 0),
        }
    except Exception as e:
        logger.warning("trading_stats_mv_query_failed", error=str(e))
        # Original real-time fallback logic could be kept here for safety
        return {"error": "Stats temporarily unavailable"}


async def get_market_statistics(db: AsyncSession, symbol: str, limit: int = 30) -> list[dict]:
    """
    Fetch pre-aggregated daily market statistics from the continuous aggregate.
    """
    try:
        result = await db.execute(
            text(
                """
                SELECT 
                    symbol,
                    day as trade_date,
                    high, low, avg_price, volume
                FROM daily_stats_chained_cagg 
                WHERE symbol = :symbol 
                ORDER BY day DESC 
                LIMIT :limit
            """
            ),
            {"symbol": symbol, "limit": limit},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("market_stats_cagg_query_failed", error=str(e))
        return []


async def get_daily_ohlcv(db: AsyncSession, symbol: str, days: int = 90) -> list[dict]:
    """
    Fetch daily stats data from TimescaleDB continuous aggregate.
    """
    try:
        # Corrected interval binding
        result = await db.execute(
            text(
                """
                SELECT 
                    day as time,
                    high, low, avg_price, volume
                FROM daily_stats_chained_cagg
                WHERE symbol = :symbol
                AND day > NOW() - (INTERVAL '1 day' * :days)
                ORDER BY day ASC
            """
            ),
            {"symbol": symbol, "days": days},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("ohlcv_cagg_query_failed", error=str(e))
        return []


async def get_iv_surface(db: AsyncSession, symbol: str, days: int = 7) -> list[dict]:
    """
    Fetch Implied Volatility surface data from continuous aggregates.
    Optimized for 3D surface plotting (Time x IV).
    """
    try:
        result = await db.execute(
            text(
                """
                SELECT 
                    bucket as time,
                    avg_iv, min_iv, max_iv, stddev_iv
                FROM iv_surface_cagg
                WHERE symbol = :symbol
                AND bucket > NOW() - (INTERVAL '1 day' * :days)
                ORDER BY bucket ASC
            """
            ),
            {"symbol": symbol, "days": days},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("iv_surface_query_failed", error=str(e))
        return []


async def get_hourly_market_stats(db: AsyncSession, symbol: str, hours: int = 24) -> list[dict]:
    """
    Fetch hourly market stats from continuous aggregates.
    """
    try:
        result = await db.execute(
            text(
                """
                SELECT 
                    hour as time,
                    avg_price, volume as total_volume, count as tick_count
                FROM hourly_stats_chained_cagg
                WHERE symbol = :symbol
                AND hour > NOW() - (INTERVAL '1 hour' * :hours)
                ORDER BY hour ASC
            """
            ),
            {"symbol": symbol, "hours": hours},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("hourly_stats_query_failed", error=str(e))
        return []


async def get_model_drift_metrics(db: AsyncSession, model_id: UUID | None = None) -> list[dict]:
    """
    Fetch pre-aggregated drift metrics from the materialized view.
    """
    try:
        query = "SELECT * FROM model_drift_metrics_mv"
        params = {}
        if model_id:
            query += " WHERE model_id = :mid"
            params["mid"] = model_id
        query += " ORDER BY window_hour DESC LIMIT 100"

        result = await db.execute(text(query), params)
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("model_drift_metrics_mv_query_failed", error=str(e))
        return []


async def get_latest_volatility_surface(db: AsyncSession, symbol: str) -> list[dict]:
    """
    Get the most recent volatility surface for a symbol from the optimized MV.
    """
    try:
        result = await db.execute(
            text("SELECT * FROM latest_vol_surface WHERE symbol = :symbol"),
            {"symbol": symbol},
        )
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("latest_vol_surface_query_failed", error=str(e))
        return []


async def get_system_health_dashboard(db: AsyncSession) -> dict:
    """
    High-Performance Diagnostic: Fetches a combined view of system performance and health.
    """
    try:
        # 1. Fetch Metrics from Continuous Aggregate
        metrics_res = await db.execute(
            text("SELECT * FROM system_metric_stats ORDER BY bucket DESC LIMIT 1")
        )
        metrics = metrics_res.first()

        # 2. Fetch Health Overview
        health_res = await db.execute(text("SELECT * FROM db_health_overview"))
        health = health_res.first()

        # 3. Fetch Cache Hit Ratio
        cache_res = await db.execute(text("SELECT * FROM cache_hit_ratio"))
        cache = cache_res.first()

        return {
            "metrics": dict(metrics._mapping) if metrics else {},
            "db_info": dict(health._mapping) if health else {},
            "cache_hit_ratio": float(cache.hit_ratio) if cache else 0.0,
        }
    except Exception as e:
        logger.error("system_health_dashboard_failed", error=str(e))
        return {"error": str(e)}


async def get_io_performance_audit(db: AsyncSession) -> list[dict]:
    """
    Fetches the new PostgreSQL 16 I/O diagnostics summary.
    """
    try:
        result = await db.execute(text("SELECT * FROM pg_stat_io_summary LIMIT 20"))
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error("io_performance_audit_failed", error=str(e))
        return []
