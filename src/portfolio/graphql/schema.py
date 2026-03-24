from collections.abc import AsyncGenerator

import strawberry
from strawberry.federation import Schema

from services.api.graphql.resolvers.portfolio_service import (
    Portfolio,
)
from services.api.graphql.resolvers.portfolio_service import (
    create_portfolio as service_create_portfolio,
)
from services.api.graphql.resolvers.portfolio_service import (
    get_portfolio as service_get_portfolio,
)
from services.api.graphql.resolvers.trading_service import (
    Order,
)
from services.api.graphql.resolvers.trading_service import (
    cancel_order as service_cancel_order,
)
from services.api.graphql.resolvers.trading_service import (
    create_order as service_create_order,
)

@strawberry.type
class Query:
    @strawberry.field
    async def portfolio(self, user_id: str) -> Portfolio | None:
        """Fetch real portfolio data from the portfolio service."""
        # Using user_id as lookup; in a real scenario, we might resolve port_id via user_id first
        # For now, we assume user_id 123 maps to port_123 for backward compatibility but using real DB
        port_id = "port_123" if user_id == "user_123" else user_id
        return await service_get_portfolio(port_id)

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_order(
        self,
        portfolio_id: strawberry.ID,
        contract_symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: float | None = None,
    ) -> Order:
        """Dispatch real order to the trading executor."""
        return await service_create_order(
            portfolio_id=portfolio_id,
            contract_symbol=contract_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )

    @strawberry.mutation
    async def cancel_order(self, order_id: strawberry.ID) -> bool:
        """Cancel an existing order via the trading executor."""
        return await service_cancel_order(order_id)

    @strawberry.mutation
    async def create_portfolio(self, user_id: str, name: str, initial_cash: float) -> Portfolio:
        """Persist a new portfolio to the database."""
        return await service_create_portfolio(user_id=user_id, name=name, initial_cash=initial_cash)

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def portfolio_updates(
        self, portfolio_id: strawberry.ID
    ) -> AsyncGenerator[Portfolio, None]:
        """
        Stream real-time portfolio updates via Redis PubSub.
        """
        from src.shared.utils.cache import get_redis

        redis = get_redis()
        if not redis:
            # Fallback to single fetch if Redis is unavailable
            port = await service_get_portfolio(str(portfolio_id))
            if port:
                yield port
            return

        pubsub = redis.pubsub()
        channel = f"portfolio_updates:{portfolio_id}"
        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    # Re-fetch from DB to ensure consistency or decode from message
                    port = await service_get_portfolio(str(portfolio_id))
                    if port:
                        yield port
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

schema = Schema(query=Query, mutation=Mutation, subscription=Subscription)
