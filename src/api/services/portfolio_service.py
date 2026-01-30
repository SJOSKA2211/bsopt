import structlog
from typing import List, Optional
from datetime import datetime
import strawberry

logger = structlog.get_logger(__name__)

@strawberry.type
class Position:
    id: strawberry.ID
    contract_symbol: str
    quantity: int
    entry_price: float

@strawberry.type
class Portfolio:
    id: strawberry.ID
    user_id: str
    cash_balance: float
    
    @strawberry.field
    async def positions(self) -> List[Position]:
        logger.debug("dummy_positions_fetch", portfolio_id=self.id)
        return [
            Position(
                id=strawberry.ID("pos_1"),
                contract_symbol="AAPL_20260115_C_150",
                quantity=10,
                entry_price=5.50
            )
        ]

async def get_portfolio(id: str) -> Optional[Portfolio]:
    logger.debug("dummy_portfolio_fetch", portfolio_id=id)
    return Portfolio(id=strawberry.ID(id), user_id="user_123", cash_balance=10000.0)

async def create_portfolio(user_id: str, name: str, initial_cash: float) -> Portfolio:
    logger.info("dummy_portfolio_create", user_id=user_id, name=name)
    return Portfolio(id=strawberry.ID("port_new"), user_id=user_id, cash_balance=initial_cash)
