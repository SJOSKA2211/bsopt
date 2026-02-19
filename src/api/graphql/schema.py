from datetime import datetime

import strawberry
from strawberry.federation import Schema


# ============================================================================
# TYPE DEFINITIONS (Base Subgraph: Options)
# ============================================================================
@strawberry.federation.type(keys=["id"])
class Option:
    """Federated Option type - provided by the Options subgraph"""

    id: strawberry.ID
    contract_symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: str

    # These fields are shared but owned by this subgraph
    # Pricing/ML/MarketData services will extend this type to add their fields

    @classmethod
    async def resolve_reference(cls, id: strawberry.ID):
        from src.api.graphql.resolvers.option_service import get_option

        return await get_option(str(id))


@strawberry.type
class Portfolio:
    id: strawberry.ID
    name: str
    cash_balance: float


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
# MUTATIONS
# ============================================================================
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_portfolio(
        self, user_id: str, name: str, initial_cash: float
    ) -> Portfolio:
        from src.api.graphql.resolvers.portfolio_service import (
            create_portfolio as create_portfolio_svc,
        )

        return await create_portfolio_svc(
            user_id=strawberry.ID(user_id), name=name, initial_cash=initial_cash
        )


# ============================================================================
# APOLLO FEDERATION - Subgraph Schema
# ============================================================================
schema = Schema(query=Query, mutation=Mutation, types=[Option, Portfolio])
