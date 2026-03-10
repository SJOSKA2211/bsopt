from datetime import date, datetime
from typing import Any

import strawberry


@strawberry.federation.type(keys=["id"], shareable=True)
class Option:
    """Federated Option type - shared across subgraphs"""

    id: strawberry.ID
    symbol: str = strawberry.federation.field(shareable=True)
    strike: float = strawberry.federation.field(shareable=True)
    expiry: date = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(name="optionType", shareable=True)

    # Market Data
    bid: float | None = strawberry.federation.field(default=None, shareable=True)
    ask: float | None = strawberry.federation.field(default=None, shareable=True)
    last: float | None = strawberry.federation.field(default=None, shareable=True)
    volume: int | None = strawberry.federation.field(default=None, shareable=True)
    open_interest: int | None = strawberry.federation.field(default=None, name="openInterest", shareable=True)

    # Greeks
    implied_volatility: float | None = strawberry.federation.field(default=None, name="iv", shareable=True)
    delta: float | None = strawberry.federation.field(default=None, shareable=True)
    gamma: float | None = strawberry.federation.field(default=None, shareable=True)
    vega: float | None = strawberry.federation.field(default=None, shareable=True)
    theta: float | None = strawberry.federation.field(default=None, shareable=True)
    rho: float | None = strawberry.federation.field(default=None, shareable=True)

    time: datetime = strawberry.federation.field(shareable=True)

    @classmethod
    async def resolve_reference(cls, id: strawberry.ID) -> "Option | None":
        from src.api.graphql.resolvers.option_service import get_option_by_id

        return await get_option_by_id(str(id))
