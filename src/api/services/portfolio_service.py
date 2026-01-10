from typing import List, Optional
from datetime import datetime
import strawberry

class Position:
    def __init__(self, id: str, portfolio_id: str, contract_symbol: str, quantity: int, entry_price: float, entry_date: datetime):
        self.id = strawberry.ID(id)
        self.portfolio_id = strawberry.ID(portfolio_id)
        self.contract_symbol = contract_symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date

class Portfolio:
    def __init__(self, id: str, user_id: str, name: str, cash_balance: float):
        self.id = strawberry.ID(id)
        self.user_id = strawberry.ID(user_id)
        self.name = name
        self.cash_balance = cash_balance

    async def positions(self) -> List[Position]:
        """Dummy positions resolver"""
        print(f"Dummy positions called for portfolio {self.id}")
        return [Position(id="pos1", portfolio_id=str(self.id), contract_symbol="SPY_CALL_400", quantity=10, entry_price=1.50, entry_date=datetime.now())]

async def get_portfolio(id: strawberry.ID) -> Optional[Portfolio]:
    """Dummy resolver for fetching a portfolio."""
    print(f"Dummy get_portfolio called for {id}")
    return Portfolio(id=id, user_id="user1", name="My Portfolio", cash_balance=10000.0)

async def create_portfolio(user_id: strawberry.ID, name: str, initial_cash: float) -> Portfolio:
    """Dummy resolver for creating a portfolio."""
    print(f"Dummy create_portfolio called for {user_id}, {name}, {initial_cash}")
    return Portfolio(id="new_portfolio_id", user_id=user_id, name=name, cash_balance=initial_cash)
