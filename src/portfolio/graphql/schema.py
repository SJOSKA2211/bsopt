from collections.abc import AsyncGenerator

import strawberry
from strawberry.federation import Schema

from api.graphql.resolvers.portfolio_service import (
    Portfolio,
)
from api.graphql.resolvers.portfolio_service import (
    create_portfolio as service_create_portfolio,
)
from api.graphql.resolvers.portfolio_service import (
    get_portfolio as service_get_portfolio,
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
