import strawberry
import structlog
from sqlalchemy import select

from src.database import get_session
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
            entry_price=float(db_pos.entry_price)
        )

@strawberry.type
class Portfolio:
    id: strawberry.ID
    user_id: str
    cash_balance: float

    @strawberry.field
    async def positions(self) -> list[Position]:
        """Fetch real positions from DB."""
        session = get_session()
        try:
            # RLS ensures we only see our own positions if session context is set
            result = session.execute(select(DBPosition).where(DBPosition.portfolio_id == self.id))
            return [Position.from_db(p) for p in result.scalars()]
        finally:
            session.close()

async def get_portfolio(id: str) -> Portfolio | None:
    """Fetch real portfolio from DB."""
    session = get_session()
    try:
        result = session.execute(select(DBPortfolio).where(DBPortfolio.id == id))
        db_port = result.scalar_one_or_none()
        if db_port:
            return Portfolio(
                id=strawberry.ID(str(db_port.id)), 
                user_id=str(db_port.user_id), 
                cash_balance=float(db_port.cash_balance)
            )
        return None
    finally:
        session.close()


async def create_portfolio(user_id: str, name: str, initial_cash: float) -> Portfolio:
    logger.info("dummy_portfolio_create", user_id=user_id, name=name)
    return Portfolio(
        id=strawberry.ID("port_new"), user_id=user_id, cash_balance=initial_cash
    )
