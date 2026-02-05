import asyncio
import random
from collections.abc import AsyncGenerator

import strawberry
from strawberry.federation import Schema


@strawberry.type
class MarketData:
    symbol: str
    last_price: float
    volume: int

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    
    @strawberry.field
    def last_price(self) -> float:
        return 15.0 + random.uniform(0, 1.0) # nosec B311

    @strawberry.field
    def volume(self) -> int:
        return random.randint(100, 10000) # nosec B311

    @classmethod
    def resolve_reference(cls, id: strawberry.ID):
        return cls(id=id)

@strawberry.type
class Query:
    @strawberry.field
    def _dummy_market(self) -> str:
        return "market"

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def market_data_stream(self, symbols: list[str]) -> AsyncGenerator[MarketData]:
        # Mock stream
        while True:
            for symbol in symbols:
                yield MarketData(
                    symbol=symbol,
                    last_price=150.0 + random.uniform(-1, 1), # nosec B311
                    volume=random.randint(100, 500) # nosec B311
                )
            await asyncio.sleep(0.1)

schema = Schema(query=Query, subscription=Subscription, types=[Option])
