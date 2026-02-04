import structlog
from typing import Optional
from datetime import datetime
import strawberry

logger = structlog.get_logger(__name__)

@strawberry.type
class Order:
    id: strawberry.ID
    portfolio_id: strawberry.ID
    contract_symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    limit_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

async def create_order(
    portfolio_id: strawberry.ID,
    contract_symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    limit_price: Optional[float] = None
) -> Order:
    logger.info("dummy_order_create", portfolio_id=portfolio_id, symbol=contract_symbol, side=side)
    return Order(
        id=strawberry.ID("order_123"),
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

async def cancel_order(order_id: strawberry.ID) -> bool:
    logger.info("dummy_order_cancel", order_id=order_id)
    return True

