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
        return 15.0 + random.uniform(0, 1.0)

    @strawberry.field
    def volume(self) -> int:
        return random.randint(100, 10000)

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
        """
        Real-time market data stream from the silicon mesh.
        OPTIMIZED: Low-latency polling of Shared Memory.
        """
        from src.shared.shm_mesh import SharedMemoryRingBuffer

        mesh = SharedMemoryRingBuffer(create=False)
        last_head = 0

        while True:
            # OPTIMIZED: Yield multiple ticks in a single window if available
            slices, new_head = mesh.read_latest_slices(last_head)
            if slices:
                for chunk in slices:
                    for tick in chunk:
                        sym = tick["symbol"].decode("ascii").strip("\x00")
                        if not symbols or sym in symbols:
                            yield MarketData(
                                symbol=sym,
                                last_price=tick["price"],
                                volume=tick["volume"],
                            )
                last_head = new_head

            # Sub-millisecond sleep to avoid CPU pinning in the resolver
            await asyncio.sleep(0.001)


schema = Schema(query=Query, subscription=Subscription, types=[Option])
