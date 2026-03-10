import strawberry
import structlog
from sqlalchemy import select

from src.database import get_async_db_context
from src.database.models import Portfolio as DBPortfolio
from src.database.models import Position as DBPosition

from src.shared.shm_mesh import GreeksMesh

logger = structlog.get_logger(__name__)
_greeks_mesh = GreeksMesh(create=False)


@strawberry.type
class Position:
    id: strawberry.ID
    contract_symbol: str
    quantity: int
    entry_price: float
    # Real-time Greeks
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    market_price: float | None = None
    unrealized_pnl: float | None = None

    @classmethod
    def from_db(cls, db_pos: DBPosition):
        pos = cls(
            id=strawberry.ID(str(db_pos.id)),
            contract_symbol=db_pos.symbol, # Mapping contract_symbol to symbol
            quantity=db_pos.quantity,
            entry_price=float(db_pos.entry_price),
        )
        
        # Enrich with SHM Greeks
        shm_greeks = _greeks_mesh.read(db_pos.symbol)
        if shm_greeks:
            pos.delta = shm_greeks["delta"]
            pos.gamma = shm_greeks["gamma"]
            pos.theta = shm_greeks["theta"]
            pos.vega = shm_greeks["vega"]
            pos.rho = shm_greeks["rho"]
            
        return pos


@strawberry.type
class Portfolio:
    id: strawberry.ID
    user_id: str
    cash_balance: float
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0

    @strawberry.field
    async def positions(self) -> list[Position]:
        """Fetch real positions from DB (Async)."""
        async with get_async_db_context() as session:
            try:
                # RLS ensures we only see our own positions if session context is set
                result = await session.execute(
                    select(DBPosition).where(DBPosition.portfolio_id == self.id)
                )
                db_positions = result.scalars().all()
                
                results = []
                for p in db_positions:
                    pos = Position.from_db(p)
                    results.append(pos)
                    
                    # Accumulate portfolio risk
                    if pos.delta is not None:
                        self.total_delta += pos.delta * pos.quantity
                    if pos.gamma is not None:
                        self.total_gamma += pos.gamma * pos.quantity
                    if pos.vega is not None:
                        self.total_vega += pos.vega * pos.quantity
                        
                return results
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
