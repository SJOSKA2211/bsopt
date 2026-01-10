import strawberry
from strawberry.federation import Schema
from typing import List, Optional, AsyncGenerator
import asyncio
import random

@strawberry.federation.type(keys=["id"], extend=True)
class Option:
    id: strawberry.ID = strawberry.federation.field(external=True)

@strawberry.type
class Position:
    id: strawberry.ID
    contract_symbol: str
    quantity: int
    entry_price: float
    
    @strawberry.field
    def option(self) -> Option:
        return Option(id=strawberry.ID(self.contract_symbol))

@strawberry.type
class Portfolio:
    id: strawberry.ID
    user_id: str
    cash_balance: float
    positions: List[Position]

@strawberry.type
class Query:
    @strawberry.field
    def portfolio(self, user_id: str) -> Optional[Portfolio]:
        # Mock data
        if user_id == "user_123":
            return Portfolio(
                id=strawberry.ID("port_123"),
                user_id=user_id,
                cash_balance=10000.0,
                positions=[
                    Position(
                        id=strawberry.ID("pos_1"),
                        contract_symbol="AAPL_20260115_C_150",
                        quantity=10,
                        entry_price=5.50
                    )
                ]
            )
        return None

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def portfolio_updates(self, portfolio_id: strawberry.ID) -> AsyncGenerator[Portfolio, None]:
        while True:
            # Mock P&L update simulation
            yield Portfolio(
                id=portfolio_id,
                user_id="user_123",
                cash_balance=10000.0 + random.uniform(-100, 100),
                positions=[]
            )
            await asyncio.sleep(0.1)

schema = Schema(query=Query, subscription=Subscription)
