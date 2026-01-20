from typing import Optional
from datetime import datetime
import strawberry

class Order:
    def __init__(self, id: str, portfolio_id: str, contract_symbol: str, side: str, quantity: int, order_type: str, status: str, limit_price: Optional[float] = None, created_at: Optional[datetime] = None):
        self.id = strawberry.ID(id)
        self.portfolio_id = strawberry.ID(portfolio_id)
        self.contract_symbol = contract_symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.status = status
        self.limit_price = limit_price
        self.created_at = created_at if created_at else datetime.now()

async def create_order(
    portfolio_id: strawberry.ID,
    contract_symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    limit_price: Optional[float] = None
) -> Order:
    """Dummy resolver for placing an option order."""
    print(f"Dummy create_order called for {portfolio_id}, {contract_symbol}, {side}, {quantity}, {order_type}, {limit_price}")
    return Order(id="order1", portfolio_id=portfolio_id, contract_symbol=contract_symbol, side=side, quantity=quantity, order_type=order_type, status="PENDING", limit_price=limit_price)

async def cancel_order(order_id: strawberry.ID) -> Order:
    """Dummy resolver for cancelling a pending order."""
    print(f"Dummy cancel_order called for {order_id}")
    return Order(id=order_id, portfolio_id="portfolio1", contract_symbol="SPY_CALL_400", side="BUY", quantity=1, order_type="LIMIT", status="CANCELLED", limit_price=1.0)
