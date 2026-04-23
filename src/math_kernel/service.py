import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Dict, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select, func, case

from src.database.models import Portfolio, Trade
from src.shared.config import config_registry

logger = logging.getLogger(__name__)

class Solver(Protocol):
    """Protocol for financial solvers (Phase 2)."""
    def compute(self, **kwargs) -> float: ...

try:
    import bsopt_core
    logger.info("Rust-core (bsopt_core) successfully linked.")
except ImportError:
    logger.warning("bsopt_core not found. Falling back to simulation mode.")
    bsopt_core = None

class BlackScholesSolver:
    """Analytical Black-Scholes solver using the Rust-core."""
    def compute(self, s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool = True) -> float:
        if bsopt_core:
            if is_call:
                return bsopt_core.bs_call(s, k, t, r, sigma)
            return bsopt_core.bs_put(s, k, t, r, sigma)
        # Fallback simulation logic (Phase 2)
        return s * 1.05 

# Register the solver
config_registry.register("bs_analytical", BlackScholesSolver())

class MathKernelService:
    """
    Service for core financial calculations.
    Strictly decoupled from transport layers (Axiom: Phase 2).
    """

    def __init__(self) -> None:
        self.solver = config_registry.get("bs_analytical")
        logger.info("MathKernelService initialized with solver: %s", type(self.solver).__name__)

    async def calculate_portfolio_value(self, portfolio_id: UUID, db: AsyncSession) -> float:
        """
        Calculates total portfolio value.
        Uses actual DB states (Zero Mock Axiom).
        """
        portfolio = await db.get(Portfolio, portfolio_id)
        if not portfolio:
            logger.error("Portfolio %s not found in persistence layer.", portfolio_id)
            raise ValueError(f"Portfolio {portfolio_id} not found")

        # Bolt Optimization: Offload value aggregation to the database
        # to prevent memory blowout with large portfolios (N+1 query problem).
        stmt = select(
            func.sum(
                case(
                    (func.lower(Trade.side) == "buy", Trade.quantity * Trade.price),
                    else_=-Trade.quantity * Trade.price
                )
            )
        ).filter(Trade.portfolio_id == portfolio_id)
        result = await db.execute(stmt)
        trade_value = result.scalar() or 0.0

        total_value = portfolio.cash + trade_value

        return round(float(total_value), 2)

    def get_risk_metrics(self, portfolio_id: UUID) -> Dict[str, Any]:
        """Provides simulated risk metrics for the portfolio."""
        return {
            "portfolio_id": str(portfolio_id),
            "greeks": {
                "delta": 0.45,
                "gamma": 0.02,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
