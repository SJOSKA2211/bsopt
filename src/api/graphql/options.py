
import strawberry
from strawberry.federation import Schema

from src.api.graphql.resolvers.option_service import get_option_by_id
from src.api.graphql.types import Option


@strawberry.type
class Query:
    @strawberry.field
    async def option(self, contract_symbol: str) -> Option | None:
        """Fetch real-time option data via the market data router."""
        return await get_option_by_id(contract_symbol)


schema = Schema(query=Query)

