import strawberry
import structlog
from sqlalchemy import select

from src.database import get_async_db_context
from src.database.models import Portfolio as DBPortfolio
from src.database.models import Position as DBPosition

logger = structlog.get_logger(__name__)


@strawberry.type
class Position:
    id: strawberry.ID
    contract_symbol: str
    quantity: int
    entry_price: float

    @classmethod
    def from_db(cls, db_pos: DBPosition):
        return cls(
            id=strawberry.ID(str(db_pos.id)),
            contract_symbol=db_pos.contract_symbol,
            quantity=db_pos.quantity,
            entry_price=float(db_pos.entry_price),
        )


@strawberry.type
class Portfolio:
    id: strawberry.ID
    user_id: str
    cash_balance: float

    @strawberry.field
    async def positions(self) -> list[Position]:
        """Fetch real positions from DB (Async)."""
        async with get_async_db_context() as session:
            try:
                # RLS ensures we only see our own positions if session context is set
                result = await session.execute(
                    select(DBPosition).where(DBPosition.portfolio_id == self.id)
                )
                return [Position.from_db(p) for p in result.scalars()]
            except Exception as e:
                logger.error("ws_fetch_positions_failed", portfolio_id=self.id, error=str(e))
                return []


async def get_portfolio(id: str) -> Portfolio | None:
    """Fetch real portfolio from DB (Async)."""
    async with get_async_db_context() as session:
        try:
            result = await session.execute(select(DBPortfolio).where(DBPortfolio.id == id))
            db_port = result.scalar_one_or_none()
            if db_port:
                return Portfolio(
                    id=strawberry.ID(str(db_port.id)),
                    user_id=str(db_port.user_id),
                    cash_balance=float(db_port.cash_balance),
                )
        except Exception as e:
            logger.error("ws_fetch_portfolio_failed", portfolio_id=id, error=str(e))
        return None


async def create_portfolio(user_id: str, name: str, initial_cash: float) -> Portfolio:
    logger.info("portfolio_create_initiated", user_id=user_id, name=name)
    # In a real High-Performance app, we'd persist this to DB here.
    return Portfolio(id=strawberry.ID("port_new"), user_id=user_id, cash_balance=initial_cash)
