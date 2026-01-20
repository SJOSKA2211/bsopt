import strawberry
from strawberry.federation import Schema
from typing import List, Optional
from datetime import datetime

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
        from src.api.services.option_service import get_option
        return await get_option(str(id))

@strawberry.type
class Portfolio:
    id: strawberry.ID
    name: str
    cash_balance: float

# ============================================================================
# QUERIES
# ============================================================================
@strawberry.type
class Query:
    """Root Query for Options subgraph"""
    @strawberry.field
    async def option(self, contract_symbol: str) -> Optional[Option]:
        """Get single option by contract symbol"""
        from src.api.services.option_service import get_option
        return await get_option(contract_symbol)
    
    @strawberry.field
    async def options(
        self,
        underlying: Optional[str] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
        expiry: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Option]:
        """Search options with filters"""
        from src.api.services.option_service import search_options
        return await search_options(
            underlying=underlying,
            min_strike=min_strike,
            max_strike=max_strike,
            expiry=expiry,
            limit=limit
        )

# ============================================================================
# MUTATIONS
# ============================================================================
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_portfolio(self, user_id: str, name: str, initial_cash: float) -> Portfolio:
        from src.api.services.portfolio_service import create_portfolio as create_portfolio_svc
        return await create_portfolio_svc(user_id=strawberry.ID(user_id), name=name, initial_cash=initial_cash)

# ============================================================================
# APOLLO FEDERATION - Subgraph Schema
# ============================================================================
schema = Schema(query=Query, mutation=Mutation, types=[Option, Portfolio])