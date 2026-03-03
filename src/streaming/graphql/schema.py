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


@strawberry.federation.type(keys=["id"], extend=True)
class Option:
    id: strawberry.ID = strawberry.federation.field(external=True)

    @strawberry.field
    def last_price(self) -> float:
        return 15.0 + random.uniform(0, 1.0)  # nosec B311

    @strawberry.field
    def bid(self) -> float:
        return 14.8 + random.uniform(0, 0.5)  # nosec B311

    @strawberry.field
    def ask(self) -> float:
        return 15.2 + random.uniform(0, 0.5)  # nosec B311

    @strawberry.field
    def volume(self) -> int:
        return random.randint(100, 10000)  # nosec B311

    @strawberry.field
    def open_interest(self) -> int:
        return random.randint(500, 50000)  # nosec B311

    @classmethod
    def resolve_reference(cls, id: strawberry.ID):
        return cls(id=id)


@strawberry.type
class Query:
    @strawberry.field
    def _dummy_market(self) -> str:
        return "market"

    @strawberry.field
    def market_data(self, symbol: str) -> MarketData:
        """Fetch latest spot price for an underlying symbol"""
        # In a real app, this would read from the SHM mesh
        return MarketData(
            symbol=symbol,
            last_price=155.0,
            volume=1500000,
        )


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
