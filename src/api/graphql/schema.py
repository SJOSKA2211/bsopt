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
class Query:
    """Root Query for Options subgraph"""

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
        first: int = 100,
        after: str | None = None,
    ) -> OptionConnection:
        """Search options with Relay-style pagination"""
        from src.api.graphql.resolvers.option_service import search_options_paginated

        results, has_next, next_cursor = await search_options_paginated(
            underlying=underlying,
            min_strike=min_strike,
            max_strike=max_strike,
            expiry=expiry,
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
