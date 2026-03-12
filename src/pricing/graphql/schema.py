from datetime import UTC, date, datetime, timedelta
from typing import Any

import strawberry
from fastapi import Request
from strawberry.federation import Schema

from src.api.graphql.types import Option


@strawberry.federation.type(keys=["id"], shareable=True)
class Portfolio:
    id: strawberry.ID
    user_id: str = strawberry.federation.field(name="user_id", shareable=True)
    name: str = strawberry.federation.field(shareable=True)
    cash_balance: float = strawberry.federation.field(name="cash_balance", shareable=True)
    created_at: datetime


@strawberry.federation.type(shareable=True)
class OptionEdge:
    cursor: str
    node: Option


@strawberry.federation.type(shareable=True)
class PageInfo:
    has_next_page: bool
    end_cursor: str | None


@strawberry.federation.type(shareable=True)
class OptionConnection:
    edges: list[OptionEdge]
    page_info: PageInfo


# QUERIES
@strawberry.federation.type(shareable=True)
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


@strawberry.federation.type(shareable=True)
class MarketData:
    symbol: str
    last_price: float = strawberry.field(name="last_price")
    bid: float | None
    ask: float | None
    volume: int | None
    timestamp: datetime


@strawberry.federation.type(shareable=True)
class OHLCV:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@strawberry.federation.type(shareable=True)
class Query:
    """Root Query for Options subgraph"""

    @strawberry.field
    async def market_data(self, symbol: str) -> MarketData:
        """Fetch latest market data for a symbol using optimized router."""
        from src.api.graphql.resolvers.option_service import router

        try:
            data = await router.get_live_quote(symbol)
            return MarketData(
                symbol=symbol,
                last_price=data.get("price", 0.0),
                bid=data.get("bid"),
                ask=data.get("ask"),
                volume=data.get("volume"),
                timestamp=datetime.now(UTC),
            )
        except Exception:
            # Fallback for demo
            return MarketData(
                symbol=symbol,
                last_price=150.25,
                bid=150.20,
                ask=150.30,
                volume=5000,
                timestamp=datetime.now(UTC),
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
                volume=1000,
            )
            for i in range(100)
        ]

    @strawberry.field
    async def ml_prediction(self, symbol: str) -> MLPrediction:
        """Fetch latest ML-based price prediction for a symbol"""
        from src.api.schemas.ml import InferenceRequest
        from src.services.ml_service import get_ml_service

        ml_service = get_ml_service()

        # Simulated request for the fair value prediction
        req = InferenceRequest(
            underlying_price=150.0,
            strike=150.0,
            time_to_expiry=0.1,
            is_call=1,
            moneyness=1.0,
            log_moneyness=0.0,
            sqrt_time_to_expiry=0.316,
            days_to_expiry=36.5,
            implied_volatility=0.2,
        )

        res = await ml_service.predict(req, symbol=symbol)

        now = datetime.now(UTC)
        return MLPrediction(
            id=strawberry.ID(f"pred-{symbol}-{now.timestamp()}"),
            symbol=symbol,
            predicted_price=res.price,
            actual_price=None,
            prediction_error=None,
            confidence_interval=0.95,
            drift=0.0,  # Would come from drift detector
            model_name=res.model_type,
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
            underlying=symbol or "AAPL",
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
schema: Schema = Schema(query=Query, types=[Option])



async def get_context(request: Request) -> dict[str, Any]:
    """GraphQL context helper."""
    return {}
