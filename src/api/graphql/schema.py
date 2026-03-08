from datetime import UTC, date, datetime, timedelta

import strawberry
from strawberry.federation import Schema


# TYPE DEFINITIONS (Base Subgraph: Options)
@strawberry.federation.type(keys=["id"], shareable=True)
class Option:
    """Federated Option type - provided by the Options subgraph"""

    id: strawberry.ID
    symbol: str = strawberry.federation.field(shareable=True)
    strike: float = strawberry.federation.field(shareable=True)
    expiry: date = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(name="optionType", shareable=True)

    # Market Data
    bid: float | None = strawberry.federation.field(shareable=True)
    ask: float | None = strawberry.federation.field(shareable=True)
    last: float | None = strawberry.federation.field(shareable=True)
    volume: int | None = strawberry.federation.field(shareable=True)
    open_interest: int | None = strawberry.federation.field(name="openInterest", shareable=True)

    # Greeks (Optimized DB data types)
    implied_volatility: float | None = strawberry.federation.field(name="iv", shareable=True)
    delta: float | None = strawberry.federation.field(shareable=True)
    gamma: float | None = strawberry.federation.field(shareable=True)
    vega: float | None = strawberry.federation.field(shareable=True)
    theta: float | None = strawberry.federation.field(shareable=True)
    rho: float | None = strawberry.federation.field(shareable=True)

    time: datetime = strawberry.federation.field(shareable=True)

    @classmethod
    async def resolve_reference(cls, id: strawberry.ID):
        from src.api.graphql.resolvers.option_service import get_option_by_id

        return await get_option_by_id(str(id))


@strawberry.federation.type(keys=["id"], shareable=True)
class Portfolio:
    id: strawberry.ID
    user_id: str = strawberry.federation.field(name="user_id", shareable=True)
    name: str = strawberry.federation.field(shareable=True)
    cash_balance: float = strawberry.federation.field(name="cash_balance", shareable=True)
    created_at: datetime


@strawberry.type
class OptionEdge:
    cursor: str
    node: Option


@strawberry.type
class PageInfo:
    has_next_page: bool
    end_cursor: str | None


@strawberry.type
class OptionConnection:
    edges: list[OptionEdge]
    page_info: PageInfo


# QUERIES
@strawberry.type
class MLPrediction:
    id: strawberry.ID
    symbol: str
    predicted_price: float = strawberry.field(name="predicted_price")
    actual_price: float | None = strawberry.field(name="actual_price")
    prediction_error: float | None = strawberry.field(name="prediction_error")
    confidence_interval: float | None = strawberry.field(name="confidence_interval")
    drift: float | None = strawberry.field(name="drift")
    model_name: str = strawberry.field(name="model_name")
    timestamp: datetime
    last_updated: datetime = strawberry.field(name="last_updated")


@strawberry.type
class MarketData:
    symbol: str
    last_price: float = strawberry.field(name="last_price")
    bid: float | None
    ask: float | None
    volume: int | None
    timestamp: datetime


@strawberry.type
class OHLCV:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@strawberry.type
class Query:
    """Root Query for Options subgraph"""

    @strawberry.field
    async def market_data(self, symbol: str) -> MarketData:
        """Fetch latest market data for a symbol"""
        return MarketData(
            symbol=symbol,
            last_price=150.25,
            bid=150.20,
            ask=150.30,
            volume=5000,
            timestamp=datetime.now(UTC)
        )

    @strawberry.field
    async def historical_data(self, symbol: str) -> list[OHLCV]:
        """Fetch historical OHLCV data for a symbol"""
        now = datetime.now(UTC)
        return [
            OHLCV(
                time=(now - timedelta(minutes=i)).isoformat(),
                open=150.0 + i,
                high=151.0 + i,
                low=149.0 + i,
                close=150.5 + i,
                volume=1000
            ) for i in range(100)
        ]

    @strawberry.field
    async def ml_prediction(self, symbol: str) -> MLPrediction:
        """Fetch latest ML-based price prediction for a symbol"""
        now = datetime.now(UTC)
        return MLPrediction(
            id=strawberry.ID("pred-123"),
            symbol=symbol,
            predicted_price=157.50,
            actual_price=None,
            prediction_error=None,
            confidence_interval=0.95,
            drift=0.02,
            model_name="XGBoost-V4-Ensemble",
            timestamp=now,
            last_updated=now,
        )

    @strawberry.field
    async def option(
        self, symbol: str, expiry: date, strike: float, option_type: str
    ) -> Option | None:
        """Get single option by primary key components"""
        from src.api.graphql.resolvers.option_service import get_option

        return await get_option(symbol, expiry, strike, option_type)

    @strawberry.field
    async def options(
        self,
        symbol: str | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
        expiry: date | None = None,
        first: int = 100,
        after: str | None = None,
    ) -> OptionConnection:
        """Search options with Relay-style pagination (Optimized Index Usage)"""
        from src.api.graphql.resolvers.option_service import search_options_paginated

        results, has_next, next_cursor = await search_options_paginated(
            symbol=symbol,
            min_strike=min_strike,
            max_strike=max_strike,
            expiry=expiry,
            limit=first,
            cursor=after,
        )

        edges = [
            OptionEdge(cursor=f"{res.symbol}_{res.expiry}_{res.strike}_{res.option_type}", node=res)
            for res in results
        ]

        return OptionConnection(
            edges=edges,
            page_info=PageInfo(has_next_page=has_next, end_cursor=next_cursor),
        )


# APOLLO FEDERATION - Subgraph Schema
schema = Schema(query=Query, types=[Option, Portfolio])
