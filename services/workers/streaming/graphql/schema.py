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
    def market_data(self, symbol: str) -> MarketData:
        """Fetch latest spot price for an underlying symbol from silicon mesh"""
        from core.shared.shm_mesh import GreeksMesh

        mesh = GreeksMesh(create=False)
        data = mesh.read(symbol)

        # Fallback to realistic defaults if SHM is empty
        price = data["delta"] * 100.0 if data else 155.0
        return MarketData(
            symbol=symbol,
            last_price=float(price),
            volume=1500000,
        )


class MeshListener:
    """
    Singleton listener for the Shared Memory Mesh.
    OPTIMIZED: Single background task polls SHM and broadcasts to all active subscribers.
    """

    _instance = None

    def __init__(self):
        self._queues: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._queues.add(queue)
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run())
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        async with self._lock:
            self._queues.discard(queue)
            if not self._queues and self._task:
                self._task.cancel()
                self._task = None

    async def _run(self):
        from core.shared.shm_mesh import SharedMemoryRingBuffer

        try:
            mesh = SharedMemoryRingBuffer(create=False)
            last_head = 0
            while True:
                ticks, new_head = mesh.read_latest_msgspec(last_head)
                if ticks:
                    async with self._lock:
                        for queue in self._queues:
                            for tick in ticks:
                                try:
                                    queue.put_nowait(tick)
                                except asyncio.QueueFull:
                                    # Drop oldest if queue is full
                                    queue.get_nowait()
                                    queue.put_nowait(tick)
                    last_head = new_head

                # Sub-millisecond adaptive sleep
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import structlog

            structlog.get_logger().error("mesh_listener_error", error=str(e))
        finally:
            if "mesh" in locals():
                mesh.close()


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def market_data_stream(self, symbols: list[str]) -> AsyncGenerator[MarketData]:
        """
        Real-time market data stream from the silicon mesh.
        OPTIMIZED: Uses shared MeshListener to minimize CPU overhead.
        """
        listener = MeshListener.get_instance()
        queue = await listener.subscribe()

        try:
            while True:
                tick = await queue.get()
                if not symbols or tick.symbol in symbols:
                    yield MarketData(
                        symbol=tick.symbol,
                        last_price=tick.price,
                        volume=tick.volume,
                    )
        finally:
            await listener.unsubscribe(queue)


schema = Schema(query=Query, subscription=Subscription, types=[Option])
