import strawberry
from strawberry.federation import Schema
from typing import List, Optional, AsyncGenerator
import asyncio
import random
from datetime import datetime

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
class Order:
    id: strawberry.ID
    portfolio_id: strawberry.ID
    contract_symbol: str
    side: str # BUY/SELL
    quantity: int
    order_type: str # LIMIT/MARKET
    status: str # PENDING/FILLED/CANCELLED
    limit_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

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
class Mutation:
    @strawberry.mutation
    async def create_order(
        self,
        portfolio_id: strawberry.ID,
        contract_symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> Order:
        # Mock order creation
        return Order(
            id=strawberry.ID(f"order_{random.randint(1000, 9999)}"),  # nosec
            portfolio_id=portfolio_id,
            contract_symbol=contract_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status="PENDING",
            limit_price=limit_price,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    @strawberry.mutation
    async def cancel_order(self, order_id: strawberry.ID) -> bool:
        return True

    @strawberry.mutation
    async def create_portfolio(self, user_id: str, name: str, initial_cash: float) -> Portfolio:
        return Portfolio(
            id=strawberry.ID(f"port_{random.randint(1000, 9999)}"),  # nosec
            user_id=user_id,
            cash_balance=initial_cash,
            positions=[]
        )

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def portfolio_updates(self, portfolio_id: strawberry.ID) -> AsyncGenerator[Portfolio, None]:
        while True:
            yield Portfolio(
                id=portfolio_id,
                user_id="user_123",
                cash_balance=10000.0 + random.uniform(-100, 100),  # nosec
                positions=[]
            )
            await asyncio.sleep(1)

schema = Schema(query=Query, mutation=Mutation, subscription=Subscription)