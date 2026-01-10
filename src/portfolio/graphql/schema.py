import strawberry
from strawberry.federation import Schema
from typing import List, Optional

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

schema = Schema(query=Query)
