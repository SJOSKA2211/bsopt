from datetime import datetime

import strawberry
from strawberry.federation import Schema


# ============================================================================
# TYPE DEFINITIONS (Base Subgraph: Options)
# ============================================================================
@strawberry.federation.type(keys=["id"], shareable=True)
class Option:
    """Federated Option type - provided by the Options subgraph"""

    id: strawberry.ID
    contract_symbol: str = strawberry.federation.field(shareable=True)
    underlying_symbol: str = strawberry.federation.field(shareable=True)
    strike: float = strawberry.federation.field(shareable=True)
    expiry: datetime = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(shareable=True)

    # These fields are shared but owned by this subgraph
    # Pricing/ML/MarketData services will extend this type to add their fields

    @classmethod
    async def resolve_reference(cls, id: strawberry.ID):
        from src.api.graphql.resolvers.option_service import get_option

        return await get_option(str(id))


@strawberry.federation.type(keys=["id"], shareable=True)
class Portfolio:
    id: strawberry.ID
    name: str = strawberry.federation.field(shareable=True)
    cash_balance: float = strawberry.federation.field(shareable=True)


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


# ============================================================================
# QUERIES
# ============================================================================
@strawberry.type
class MLPrediction:
    symbol: str
    predicted_price: float
    confidence_interval: list[float]
    drift: float
    model_name: str
    last_updated: datetime

@strawberry.type
class Query:
    """Root Query for Options subgraph"""

    @strawberry.field
    async def ml_prediction(self, symbol: str) -> MLPrediction:
        """Fetch ML-based price prediction for a symbol"""
        # In a real app, this would call MLService
        from datetime import datetime
        return MLPrediction(
            symbol=symbol,
            predicted_price=157.50,
            confidence_interval=[154.20, 160.80],
            drift=0.015,
            model_name="XGBoost-V4-Ensemble",
            last_updated=datetime.now(),
        )
    @strawberry.field
    async def option(self, contract_symbol: str) -> Option | None:
        """Get single option by contract symbol"""
        from src.api.graphql.resolvers.option_service import get_option

        return await get_option(contract_symbol)

    @strawberry.field
    async def options(
        self,
        underlying: str | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
        expiry: datetime | None = None,
        expiry_bucket: str | None = None,
        first: int = 100,
        after: str | None = None,
    ) -> OptionConnection:
        """Search options with Relay-style pagination"""
        from src.api.graphql.resolvers.option_service import search_options_paginated

        # If expiry_bucket is provided, it takes precedence over exact expiry
        results, has_next, next_cursor = await search_options_paginated(
            underlying=underlying,
            min_strike=min_strike,
            max_strike=max_strike,
            expiry=expiry,
            expiry_bucket=expiry_bucket,
            limit=first,
            cursor=after,
        )

        edges = [OptionEdge(cursor=res.id, node=res) for res in results]

        return OptionConnection(
            edges=edges,
            page_info=PageInfo(has_next_page=has_next, end_cursor=next_cursor),
        )


# ============================================================================
# APOLLO FEDERATION - Subgraph Schema
# ============================================================================
schema = Schema(query=Query, types=[Option, Portfolio])
