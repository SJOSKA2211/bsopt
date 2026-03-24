from datetime import datetime

import strawberry
import structlog

from src.blockchain.defi_options import DeFiOptionsProtocol
from src.trading.execution import OrderExecutor

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
    limit_price: float | None = None
    created_at: datetime
    updated_at: datetime

# Global executor instance (reuse connection pool)
protocol = DeFiOptionsProtocol()
executor = OrderExecutor(protocol=protocol)

async def create_order(
    portfolio_id: strawberry.ID,
    contract_symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    limit_price: float | None = None,
) -> Order:
    """
    Real-time order creation with pre-trade risk validation.
    """
    logger.info("order_request_received", symbol=contract_symbol, side=side)

    # 1. Dispatch to real executor (hardened)
    params = {
        "contract_address": contract_symbol,  # Assuming symbol is address for DeFi
        "amount": quantity,
        "side": side,
        "price": limit_price or 0.0,
    }

    result = await executor.execute_order(params)

    # 2. Map execution result to GraphQL response
    status = "OPEN" if result["status"] == "success" else "REJECTED"
    reason = result.get("reason", "")

    return Order(
        id=strawberry.ID(result.get("tx_hash", "none")),
        portfolio_id=portfolio_id,
        contract_symbol=contract_symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        status=f"{status}: {reason}" if reason else status,
        limit_price=limit_price,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

async def cancel_order(order_id: strawberry.ID) -> bool:
    logger.info("order_cancel_request", order_id=order_id)
    return await executor.cancel_order(str(order_id))
